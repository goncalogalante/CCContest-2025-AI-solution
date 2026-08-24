# Cloudflight Coding Contest - Data & AI: Migratory Birds

Full solutions for all six levels of the Cloudflight Coding Contest Data & AI track, "Migratory Birds" edition. Every level passed.

The repo contains the solvers, the original task descriptions and inputs, and the submitted answer files, so every result can be reproduced end to end.

> Task descriptions, artwork and input data are (c) Cloudflight GmbH and are included here for reference. The code in `src/` is mine.

## Contents

```
levels/            task PDF + input archive per level, as downloaded from the contest
data/level_N/      inputs extracted from levels/level_N.zip
data/parquet/      inputs converted to Parquet (regenerated on demand)
src/level_N.py     one solver per level
out/               the files submitted to the grader
```

## Setup

```bash
pip install pandas pyarrow scikit-learn scipy
python src/level_1.py     # each solver reads data/, writes out/
```

Every solver converts its input to Parquet under `data/parquet/` first and reads from there. Level 5's panel is a 58 MB CSV, 1.9 M rows, that loads in about 1.5 s once converted.

Runtimes: levels 1, 2, 3 and 6 finish in seconds; level 4 takes about a minute; level 5 a few minutes.

## Results

| level | task | metric | required | achieved |
|---|---|---|---|---|
| 1 | sort observation points | exact match | exact | pass |
| 2 | predict missing scores | RMSE | < 3 | **2.58** (CV) |
| 3 | name 6 flocks | exact match | exact | pass |
| 4 | classify 360 flocks | F1 macro | > 0.90 | **0.978** (CV) |
| 5 | 30-day top-50 forecast | daily accuracy | > 0.50 | **0.638** (backtest) |
| 6 | one day, exactly | accuracy | 100% | pass |

## The levels

**1. The Bird Observation Points.** Rank observation points (BOPs): birds prefer warmth, then low humidity, ties by lower id. The catch is in the data, not the logic - two rows spell the temperature out in English (`seventeen`), which breaks a plain integer cast.

**2. The Bird Love Score.** Predict a fifth of the scores, which are missing. Two things mattered. Some temperatures were recorded in Fahrenheit; the values fall into two clean clusters, C up to 45 and F from 70 up with nothing between, so the repair is unambiguous. And the target is almost purely temperature (r = 0.99), so plain linear regression beat boosting, polynomials and splines - residuals are clean Gaussian noise. File `c` has no labels at all, so training pools files `a` and `b`.

**3. The Hurracurra Bird.** Six flocks, six species, one of each. Match each ornithologist's prose description to a movement pattern:

| species | signature |
|---|---|
| Medieval Bluetit | palindromic route, "flies back along the same route" |
| Sticky Wolfthroat | also palindromic, but a *subset* of a Bluetit route - it ambushes them |
| Flanking Blackfinch | identical closed loop for every bird, "orbits a chosen land" |
| Rusty Goldhammer | shared outbound leg, then splits for "me-time" |
| Red Firefinch | shared nest only, diverging immediately |
| Hurracurra Bird | free shape, but its worms "only live in hot regions" |

Pure structural rules, no model. The subset test resolves the two palindromic species, and temperature separates the last two by a wide margin: 43.6 C against 22.6 for the next highest.

**4. Going Global.** Same question at scale: 360 flocks, 72 unlabelled, noisier paths, each judged alone. Each flock collapses to one feature row describing its birds' paths, then a random forest. The interesting part is Goldhammer vs Wolfthroat, which produce the same shape and differ only in path length. What really separates them is the predator relationship, since Wolfthroats wait *on Bluetit routes* - and Bluetits are trivially found by their palindromes. Overlap with a Bluetit route: 0.44 for Wolfthroat, 0.13 for Bluetit itself, under 0.09 for everyone else. That one feature carries the score.

**5. A Chirp Disaster.** 730 days x 2500 BOPs; predict the top 50 arrivals for each of the next 30 days. Arrivals are unknown throughout, but the occupancy forecast is given and tracks arrivals at ~0.86 correlation daily.

The trap is validation. Accuracy degrades steadily with recency - recent windows score 0.42 where earlier ones score 0.88 - so a model tuned on the full history looks great and then underperforms on the real target. Validating on recent windows only is what fixed it.

The final version scores what is actually graded, *is this BOP in today's top 50*, with a classifier, rather than regressing arrivals and ranking them. All the loss sits at the boundary, so training on the boundary is the point. Its probability is rank-averaged 60/40 with the occupancy rank, the one signal that cannot go stale. Backtest on the five most recent windows: mean 0.638, worst 0.623, where earlier approaches all had a window in the 0.42-0.56 range.

Two findings that didn't reach the final model but are real: departures on a day are almost exactly the previous day's arrivals (r = 0.945), which collapses the bookkeeping identity to `arrivals = occupancy + per-BOP offset`; and the level 4 path graph really is the movement mechanism, with neighbour inflow at 0.94 correlation and fitted edge weights giving one-step R2 = 0.80, but it decays too fast to carry 30 days. Also worth knowing: the top-50 boundary is intrinsically tight, with 60 to 90 BOPs within 10% of the rank-50 value on a typical day, which is why realistic accuracy sits in the 0.5-0.7 band.

**6. Final Reckoning.** One day, 100% accuracy required. A villain switches on a "WeatherChanger" and the birds act strangely. They don't - the recording is playing backwards. Occupancy on new day `t` is bit-identical to level 5's day `1521 - t` for all 2500 BOPs on all 30 days, and the wind is exactly negated.

The subtlety is which mirrored quantity the arrivals column holds. A physically time-reversed run would report the old day's *departures*, shifting the answer by a day and changing 20% of the top 50. That is ruled out because the departures implied by the new series go negative 1955 times, while real departures are never negative (0 of 77 500) - a genuine reversed simulation cannot do that, a replayed recording can. So the target day mirrors the last day with known arrivals and the answer is read straight out of the level 5 data, no model at all. The solver asserts the mirror on all 30 days before relying on it.

## Notes

- The Fahrenheit temperatures reappear in every level that reuses the level 1 table, and numbers are occasionally spelled out in words.
- `data/level_6/all_data_from_level_5.in` is the level 5 panel shipped again inside the level 6 archive; the two 58 MB files are duplicates.
- `data/parquet/` is generated and can be deleted; each solver rebuilds what it needs.
