"""Level 2 - predict the missing Bird Love Scores.

Join the level 1 temperature/humidity table, repair the temperatures reported
in Fahrenheit, then fit a linear model on the labelled rows (files a + b).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "level_2"
PARQUET_DIR = ROOT / "data" / "parquet"
OUT_DIR = ROOT / "out"

FEATURES = ["veg", "ins", "light", "temp", "hum"]
TARGET_HEADER = "BOP,Bird Love Score [<3]"

# The level 1 temperatures top out at 45 C and the Fahrenheit ones start at 70,
# so there is a clean gap to split on.
CELSIUS_MAX = 45


def load_level_1() -> pd.DataFrame:
    df = pd.read_csv(IN_DIR / "all_data_from_level_1.in", encoding="utf-8")
    df.columns = ["bop", "temp", "hum"]
    df["temp"] = np.where(
        df["temp"] > CELSIUS_MAX, (df["temp"] - 32) * 5 / 9, df["temp"].astype(float)
    )
    return df


def to_parquet(in_path: Path, level_1: pd.DataFrame) -> Path:
    df = pd.read_csv(in_path, encoding="utf-8")
    df.columns = ["bop", "veg", "ins", "light", "bls"]
    df["bls"] = pd.to_numeric(df["bls"], errors="coerce")  # "missing" -> NaN
    df = df.merge(level_1, on="bop", how="left", validate="one_to_one")

    parquet_path = PARQUET_DIR / f"{in_path.stem}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return parquet_path


def main() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    level_1 = load_level_1()
    tables = {
        path.stem: pd.read_parquet(to_parquet(path, level_1), engine="pyarrow")
        for path in sorted(IN_DIR.glob("level_2_*.in"))
    }

    graded = pd.concat([tables[k] for k in ("level_2_a", "level_2_b")])
    train = graded.dropna(subset=["bls"])

    score = -cross_val_score(
        LinearRegression(),
        train[FEATURES],
        train["bls"],
        cv=KFold(5, shuffle=True, random_state=0),
        scoring="neg_root_mean_squared_error",
    )
    print(f"5-fold CV RMSE: {score.mean():.3f} (+/- {score.std():.3f})")

    model = LinearRegression().fit(train[FEATURES], train["bls"])
    print("coefficients:", dict(zip(FEATURES, model.coef_.round(4))))

    for stem, df in tables.items():
        missing = df[df["bls"].isna()]
        if missing.empty:
            continue
        prediction = model.predict(missing[FEATURES])
        lines = [TARGET_HEADER] + [
            f"{bop},{value:.2f}" for bop, value in zip(missing["bop"], prediction)
        ]
        out_path = OUT_DIR / f"{stem}.csv"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{stem}: {len(missing)} predictions -> {out_path.name}")


if __name__ == "__main__":
    main()
