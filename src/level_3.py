"""Level 3 - name each of the 6 flocks from the shape of its BOP paths.

Bluetit: palindrome route. Wolfthroat: palindrome contained in the Bluetit's.
Blackfinch: every bird flies the same closed loop. Goldhammer: shared outbound
leg, then splits. Hurracurra: hot regions. Firefinch: shared nest only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "level_3"
PARQUET_DIR = ROOT / "data" / "parquet"
OUT_DIR = ROOT / "out"

CELSIUS_MAX = 45  # above this a level 1 temperature is really Fahrenheit


def load_temperatures() -> dict[int, float]:
    df = pd.read_csv(IN_DIR / "all_data_from_level_1.in", encoding="utf-8")
    df.columns = ["bop", "temp", "hum"]
    df["temp"] = np.where(
        df["temp"] > CELSIUS_MAX, (df["temp"] - 32) * 5 / 9, df["temp"].astype(float)
    )
    return dict(zip(df["bop"], df["temp"]))


def to_parquet(in_path: Path, temps: dict[int, float]) -> Path:
    raw = pd.read_csv(in_path, encoding="utf-8")
    raw.columns = ["flock", "path"]
    paths = raw["path"].map(lambda s: [int(x) for x in s.split()])

    df = pd.DataFrame(
        {
            "flock": raw["flock"].astype("int64"),
            "path": paths,
            "length": paths.map(len),
            "is_palindrome": paths.map(lambda p: p == p[::-1]),
            "mean_temp": paths.map(lambda p: float(np.mean([temps[b] for b in p]))),
        }
    )
    parquet_path = PARQUET_DIR / f"{in_path.stem}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return parquet_path


def shared_prefix(paths: list[list[int]]) -> int:
    first = paths[0]
    for i, bop in enumerate(first):
        if any(len(p) <= i or p[i] != bop for p in paths):
            return i
    return len(first)


def classify(df: pd.DataFrame) -> dict[int, str]:
    flocks = {
        int(flock): group["path"].map(list).tolist()
        for flock, group in df.groupby("flock", sort=True)
    }
    bops = {f: {b for p in ps for b in p} for f, ps in flocks.items()}
    mean_temp = df.groupby("flock")["mean_temp"].mean()

    species: dict[int, str] = {}

    # The two palindromic flocks: the one contained in the other waits along
    # its route, so it is the bird of prey.
    palindromic = [f for f, ps in flocks.items() if all(p == p[::-1] for p in ps)]
    bluetit = max(palindromic, key=lambda f: len(bops[f]))
    (wolfthroat,) = [f for f in palindromic if f != bluetit]
    assert bops[wolfthroat] <= bops[bluetit], "wolfthroat should hunt on the route"
    species[bluetit] = "Medieval Bluetit"
    species[wolfthroat] = "Sticky Wolfthroat"

    rest = [f for f in flocks if f not in species]
    for flock in rest:
        paths = flocks[flock]
        if len({tuple(p) for p in paths}) == 1:
            species[flock] = "Flanking Blackfinch"
        elif shared_prefix(paths) > 1:
            species[flock] = "Rusty Goldhammer"

    # The two free-form flocks are told apart by the worms' hot regions.
    free = sorted([f for f in flocks if f not in species], key=lambda f: -mean_temp[f])
    hurracurra, firefinch = free
    species[hurracurra] = "Hurracurra Bird"
    species[firefinch] = "Red Firefinch"

    assert len(set(species.values())) == 6, f"ambiguous assignment: {species}"
    return species


def main() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temps = load_temperatures()

    for in_path in sorted(IN_DIR.glob("level_3_*.in")):
        if in_path.stem.endswith(("species", "sample")):
            continue  # the sample has only 2 of the 6 flocks
        df = pd.read_parquet(to_parquet(in_path, temps), engine="pyarrow")
        species = classify(df)

        lines = ["Flock ID,Species"] + [
            f"{flock},{species[flock]}" for flock in sorted(species)
        ]
        out_path = OUT_DIR / f"{in_path.stem}.csv"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{in_path.stem}: {species}")


if __name__ == "__main__":
    main()
