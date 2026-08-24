"""Level 1 - sort BOPs by popularity: temperature desc, humidity asc, id asc."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "level_1"
PARQUET_DIR = ROOT / "data" / "parquet"
OUT_DIR = ROOT / "out"

COLUMNS = ["bop", "temperature", "humidity"]

# Some rows spell their number out in English ("seventeen" instead of 17).
UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def to_int(value: str) -> int:
    """Parse an integer that may be written in digits or in English words."""
    text = str(value).strip().lower().replace("-", "-")
    try:
        return int(text)
    except ValueError:
        pass

    sign = 1
    for prefix in ("minus ", "negative ", "-"):
        if text.startswith(prefix):
            sign, text = -1, text[len(prefix):].strip()
            break

    total = 0
    for word in text.replace("-", " ").replace(" and ", " ").split():
        if word in UNITS:
            total += UNITS[word]
        elif word in TENS:
            total += TENS[word]
        elif word == "hundred":
            total = (total or 1) * 100
        else:
            raise ValueError(f"cannot parse number: {value!r}")
    return sign * total


def to_parquet(in_path: Path) -> Path:
    df = pd.read_csv(
        in_path, encoding="utf-8", skiprows=1, names=COLUMNS, dtype=str
    )
    for column in COLUMNS:
        df[column] = df[column].map(to_int).astype("int64")
    parquet_path = PARQUET_DIR / f"{in_path.stem}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return parquet_path


def solve(parquet_path: Path) -> str:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.sort_values(
        ["temperature", "humidity", "bop"],
        ascending=[False, True, True],
        kind="stable",
    )
    return " ".join(df["bop"].astype(str))


def main() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for in_path in sorted(IN_DIR.glob("level_1_*.in")):
        answer = solve(to_parquet(in_path))
        out_path = OUT_DIR / f"{in_path.stem}.out"
        out_path.write_text(answer + "\n", encoding="utf-8")
        print(f"{in_path.name} -> {out_path.name}: {answer[:60]}...")

    expected = (IN_DIR / "level_1_sample.out").read_text(encoding="utf-8").strip()
    got = (OUT_DIR / "level_1_sample.out").read_text(encoding="utf-8").strip()
    print("sample:", "OK" if expected == got else f"FAIL expected={expected!r} got={got!r}")


if __name__ == "__main__":
    main()
