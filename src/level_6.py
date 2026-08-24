"""Level 6 - the top 50 Arrivals BOPs for day 791, exactly.

The WeatherChanger just replays the old recording backwards: occupancy on new
day t is bit-identical to level 5's day 1521 - t, and the wind is negated. The
Arrivals column is the mirrored day's arrivals, not its departures - the
departures implied by the new series go negative 1955 times, which a real
reversed run could not do. So day 791 mirrors day 730 and the answer is read
straight out of the level 5 data.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "level_6"
PARQUET_DIR = ROOT / "data" / "parquet"
OUT_DIR = ROOT / "out"

TARGET_DAY = 791
MIRROR_SUM = 1521  # new day t mirrors old day MIRROR_SUM - t
TOP_K = 50


def to_parquet(name: str, source: Path) -> pd.DataFrame:
    parquet_path = PARQUET_DIR / f"{name}.parquet"
    if not parquet_path.exists():
        raw = pd.read_csv(source, encoding="utf-8", low_memory=False)
        raw.columns = ["day", "bop", "arrivals", "occupancy", "wx", "wy", "ins"]
        raw["arrivals"] = pd.to_numeric(raw["arrivals"], errors="coerce")
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(parquet_path, engine="pyarrow", index=False)
    return pd.read_parquet(parquet_path, engine="pyarrow")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = to_parquet("level_5", IN_DIR / "all_data_from_level_5.in")
    current = to_parquet("level_6", IN_DIR / "level_6.in")

    old_occupancy = previous.pivot(index="day", columns="bop", values="occupancy")
    new_occupancy = current.pivot(index="day", columns="bop", values="occupancy")
    old_arrivals = previous.pivot(index="day", columns="bop", values="arrivals")

    # Verify the mirror on every day we can check before relying on it.
    mismatched = [
        day
        for day in new_occupancy.index
        if not (new_occupancy.loc[day].values == old_occupancy.loc[MIRROR_SUM - day].values).all()
    ]
    assert not mismatched, f"occupancy mirror broken on days {mismatched}"
    print(f"occupancy mirror verified on all {len(new_occupancy.index)} new days")

    source_day = MIRROR_SUM - TARGET_DAY
    ranking = old_arrivals.loc[source_day].nlargest(TOP_K)
    print(f"day {TARGET_DAY} mirrors day {source_day}")

    line = f"{TARGET_DAY}," + " ".join(str(int(bop)) for bop in ranking.index)
    out_path = OUT_DIR / "level_6.csv"
    out_path.write_text("Day,Top 50 Arrivals BOPs\n" + line + "\n", encoding="utf-8")
    print(f"-> {out_path}")
    print(line[:110] + " ...")


if __name__ == "__main__":
    main()
