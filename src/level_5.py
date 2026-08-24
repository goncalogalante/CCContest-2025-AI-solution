"""Level 5 - the 50 BOPs with the highest Arrivals for days 731-760.

Occupancy is given for every forecast day and carries the signal (~0.86
correlation with Arrivals). A classifier scores what is actually graded - is
this BOP in the day's top 50 - and its probability is rank-averaged with the
raw occupancy rank. Backtest on the five most recent windows: 0.638 mean,
0.623 worst; the level needs 0.50.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "level_5"
PARQUET_DIR = ROOT / "data" / "parquet"
OUT_DIR = ROOT / "out"

LAST_KNOWN_DAY = 730
HORIZON = 30
TOP_K = 50
OCCUPANCY_WEIGHT = 0.4
TRAIN_CUTS = list(range(300, LAST_KNOWN_DAY - HORIZON + 1, 20))

FEATURES = [
    "h",
    "occ",
    "occ_rank",
    "r60",
    "r20",
    "offset",
    "pred_ratio",
    "pred_offset",
    "docc",
    "docc7",
    "occ_vs_cut",
    "arrivals_cut",
    "arrivals_cut_rank",
    "arrivals_mean30",
]


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    parquet_path = PARQUET_DIR / "level_5.parquet"
    if not parquet_path.exists():
        raw = pd.read_csv(IN_DIR / "level_5.in", encoding="utf-8", low_memory=False)
        raw.columns = ["day", "bop", "arrivals", "occupancy", "wx", "wy", "ins"]
        raw["arrivals"] = pd.to_numeric(raw["arrivals"], errors="coerce")
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(parquet_path, engine="pyarrow", index=False)
    panel = pd.read_parquet(parquet_path, engine="pyarrow")

    arrivals = panel.pivot(index="day", columns="bop", values="arrivals")
    occupancy = panel.pivot(index="day", columns="bop", values="occupancy")
    return arrivals.loc[:LAST_KNOWN_DAY], occupancy


def day_features(arrivals, occupancy, cut: int, day: int) -> pd.DataFrame:
    """Everything knowable at `cut` about `day`, plus that day's occupancy."""
    window = arrivals.loc[cut - 60 : cut] / occupancy.loc[cut - 60 : cut]
    ratio60 = window.replace([np.inf, -np.inf], np.nan).mean()
    window20 = arrivals.loc[cut - 20 : cut] / occupancy.loc[cut - 20 : cut]
    ratio20 = window20.replace([np.inf, -np.inf], np.nan).mean()
    # arrivals_t = occ_t + offset, from the one-day-residence identity
    offset = (arrivals.loc[cut - 59 : cut] - occupancy.loc[cut - 59 : cut]).mean()
    occ = occupancy.loc[day]

    return pd.DataFrame(
        {
            "bop": arrivals.columns.values,
            "day": day,
            "h": day - cut,
            "occ": occ.values,
            "occ_rank": occ.rank(pct=True).values,
            "r60": ratio60.values,
            "r20": ratio20.values,
            "offset": offset.values,
            "pred_ratio": (occ * ratio60).rank(pct=True).values,
            "pred_offset": (occ + offset).rank(pct=True).values,
            "docc": (occ - occupancy.loc[day - 1]).values,
            "docc7": (occ - occupancy.loc[day - 7]).values,
            "occ_vs_cut": (occ / (occupancy.loc[cut] + 1)).values,
            "arrivals_cut": arrivals.loc[cut].values,
            "arrivals_cut_rank": arrivals.loc[cut].rank(pct=True).values,
            "arrivals_mean30": arrivals.loc[cut - 29 : cut].mean().values,
        }
    )


def window_frame(arrivals, occupancy, cut: int, labelled: bool) -> pd.DataFrame:
    frames = []
    for day in range(cut + 1, cut + 1 + HORIZON):
        frame = day_features(arrivals, occupancy, cut, day)
        if labelled:
            top = set(arrivals.loc[day].nlargest(TOP_K).index)
            frame["y"] = frame["bop"].isin(top)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrivals, occupancy = load_panel()

    train = pd.concat(
        [window_frame(arrivals, occupancy, cut, True) for cut in TRAIN_CUTS],
        ignore_index=True,
    )
    print(f"training rows: {len(train)} over {len(TRAIN_CUTS)} windows")

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.07, random_state=0
    )
    model.fit(train[FEATURES], train["y"])

    future = window_frame(arrivals, occupancy, LAST_KNOWN_DAY, False)
    future["p"] = model.predict_proba(future[FEATURES])[:, 1]
    future["score"] = OCCUPANCY_WEIGHT * future.groupby("day")["occ"].rank(pct=True) + (
        1 - OCCUPANCY_WEIGHT
    ) * future.groupby("day")["p"].rank(pct=True)

    lines = ["Day,Top 50 Arrivals BOPs"]
    for day in range(LAST_KNOWN_DAY + 1, LAST_KNOWN_DAY + 1 + HORIZON):
        picked = future[future["day"] == day].nlargest(TOP_K, "score")["bop"]
        lines.append(f"{day}," + " ".join(str(int(bop)) for bop in picked))

    out_path = OUT_DIR / "level_5.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 1} days -> {out_path}")
    print(lines[1][:100] + " ...")


if __name__ == "__main__":
    main()
