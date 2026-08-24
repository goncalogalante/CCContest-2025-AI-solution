"""Level 4 - classify 360 flocks, 72 of them unlabelled.

One feature row per flock describing the shape of its birds' paths, then a
random forest. The decisive feature is how much of a flock's territory lies on
a Bluetit route, which is what separates Wolfthroat from Goldhammer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "level_4"
PARQUET_DIR = ROOT / "data" / "parquet"
OUT_DIR = ROOT / "out"

CELSIUS_MAX = 45
MISSING = "missing"


def load_temperatures() -> dict[int, float]:
    df = pd.read_csv(IN_DIR / "all_data_from_level_1.in", encoding="utf-8")
    df.columns = ["bop", "temp", "hum"]
    df["temp"] = np.where(
        df["temp"] > CELSIUS_MAX, (df["temp"] - 32) * 5 / 9, df["temp"].astype(float)
    )
    return dict(zip(df["bop"], df["temp"]))


def longest_common_substring(a: list[int], b: list[int]) -> int:
    row = [0] * (len(b) + 1)
    best = 0
    for x in a:
        nxt = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            if x == y:
                nxt[j + 1] = row[j] + 1
                best = max(best, nxt[j + 1])
        row = nxt
    return best


def common_prefix(paths: list[list[int]]) -> int:
    for i, bop in enumerate(paths[0]):
        if any(len(p) <= i or p[i] != bop for p in paths):
            return i
    return len(paths[0])


def flock_features(paths: list[list[int]], temps: dict[int, float]) -> dict:
    sets = [set(p) for p in paths]
    lengths = [len(p) for p in paths]
    mean_len = float(np.mean(lengths))
    pairs = [(a, b) for i, a in enumerate(paths) for b in paths[i + 1 :]]
    jaccard = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in pairs] or [1.0]
    substrings = [
        longest_common_substring(a, b) / min(len(a), len(b)) for a, b in pairs
    ] or [1.0]
    shortest = min(lengths)
    aligned = [i for i in range(shortest) if len({p[i] for p in paths}) == 1]
    path_temps = [temps[b] for p in paths for b in p]

    return {
        "n_birds": len(paths),
        "mean_len": mean_len,
        "std_len": float(np.std(lengths)),
        "n_bops": len(set().union(*sets)),
        # Bluetits fly home the way they came.
        "palindrome": float(np.mean([p == p[::-1] for p in paths])),
        "revisit": float(np.mean([len(p) / len(set(p)) for p in paths])),
        "closed": float(np.mean([p[0] == p[-1] for p in paths])),
        # How much the birds stay together.
        "identical": float(len({tuple(p) for p in paths}) == 1),
        "distinct_frac": len({tuple(p) for p in paths}) / len(paths),
        "jaccard": float(np.mean(jaccard)),
        "jaccard_max": float(np.max(jaccard)),
        "jaccard_min": float(np.min(jaccard)),
        "substring": float(np.mean(substrings)),
        "substring_max": float(np.max(substrings)),
        "intersect_frac": len(set.intersection(*sets))
        / float(np.mean([len(s) for s in sets])),
        # A shared outbound leg that later splits up (Goldhammer).
        "prefix_frac": common_prefix(paths) / mean_len,
        "aligned_frac": len(aligned) / mean_len,
        "aligned_pos": float(np.mean(aligned)) / mean_len if aligned else -1.0,
        "aligned_spread": float(np.std(aligned)) / mean_len if len(aligned) > 1 else 0.0,
        # Hurracurra worms live in hot regions.
        "temp_mean": float(np.mean(path_temps)),
        "temp_min": float(np.min(path_temps)),
        "temp_max": float(np.max(path_temps)),
    }


def build_table(temps: dict[int, float]) -> pd.DataFrame:
    raw = pd.read_csv(IN_DIR / "level_4.in", encoding="utf-8")
    raw.columns = ["flock", "path", "species"]
    raw["path"] = raw["path"].map(lambda s: [int(x) for x in s.split()])

    flocks = {int(f): g["path"].tolist() for f, g in raw.groupby("flock")}
    labels = raw.groupby("flock")["species"].first()
    bops = {f: set().union(*[set(p) for p in ps]) for f, ps in flocks.items()}

    # Bluetits are identifiable from their palindromes alone, and the
    # Wolfthroats hunt along their routes - so containment in a Bluetit
    # territory is what separates the two "shared outbound leg" species.
    bluetit_like = [
        f for f, ps in flocks.items() if np.mean([p == p[::-1] for p in ps]) > 0.5
    ]
    rows = []
    for flock, paths in flocks.items():
        mine = bops[flock]
        others = [bops[g] for g in bluetit_like if g != flock]
        rows.append(
            {
                "flock": flock,
                "species": labels[flock],
                **flock_features(paths, temps),
                "bluetit_overlap": max(
                    (len(mine & other) / len(mine) for other in others), default=0.0
                ),
                "bluetit_overlap_all": (
                    len(mine & set().union(*others)) / len(mine) if others else 0.0
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("flock").reset_index(drop=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_DIR / "level_4.parquet", engine="pyarrow", index=False)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table(load_temperatures())

    features = [c for c in df.columns if c not in ("flock", "species")]
    train = df[df["species"] != MISSING]
    test = df[df["species"] == MISSING]

    model = RandomForestClassifier(1000, random_state=0)
    predicted = cross_val_predict(
        model,
        train[features],
        train["species"],
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
    )
    print(f"CV F1 macro: {f1_score(train['species'], predicted, average='macro'):.4f}")
    print(classification_report(train["species"], predicted, digits=3))

    model.fit(train[features], train["species"])
    guesses = model.predict(test[features])

    lines = ["Flock ID,Species"] + [
        f"{flock},{species}" for flock, species in zip(test["flock"], guesses)
    ]
    out_path = OUT_DIR / "level_4.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(test)} predictions -> {out_path}")
    print(pd.Series(guesses).value_counts().to_string())


if __name__ == "__main__":
    main()
