[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/junod1908/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/junod1908/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

An entry to the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
a program (`main.py`, Python 3.13, standard library only) that reads one case — every read the
house made of Exophase's Grand Theft Auto VI (PS5) trophy page before the case's checkpoint — and
writes five quantiles (5/25/50/75/95%) of each trophy's earn rate at the case's target, scored by
mean pinball loss on the logit scale.

## Why not persistence

The page is a step function. Exophase refreshes a game's counts on its own schedule, hours apart:
between refreshes every read is identical, and at a refresh the tracked-player count jumps and
every rate moves at once, because the newly tracked players hold fewer trophies than the cohort
tracked before them. The house persistence baseline says "rates stay put with a narrow random-walk
spread"; on the practice that scores about 1.51, because the first refresh took every rate outside
its band. This program models the two stocks that make a rate, separately:

- **Players tracked** `P(t)`: a damped log-linear fit over the case's reads. If the page never
  changed during the history (a stale page — the practice case), no growth is observable, so a
  structural prior stands in: a first refresh inside the horizon is expected to multiply the
  tracked players several-fold (the launch-night refresh the rules describe took the page from
  638 to 3,868 players).
- **Trophies earned** `E_i(t) = rate_i · P / 100`: at each observed player jump, the earned-count
  rise splits into the old cohort earning more (organic growth `g_i`) and the new cohort's
  holdings (share `c_i` of the old cohort's rate). The two are fitted jointly by alternating
  least squares over the case's jumps and shrunk to priors. On a stale page the prior for `c_i`
  is shaped by the rate itself (`c_i ~ rate^0.85`): everyone holds the boot trophy, almost no
  newcomer holds a deep-progress one.

The median grows the tracked cohort's earned count at `g_i`, capped by the saturating hazard form
of the same growth, and adds the diluted contribution of the players yet to be tracked. The spread
is a logit-scale sigma that grows with the horizon and is much wider when the page is stale (a
refresh of unknown size is pending) or a trophy is very rare (its logit swings furthest when the
denominator jumps).

## Validation (honest)

The practice answers are not in the kit, so the practice was validated against a family of
dilution hypotheses for the refreshed page (players 638 → 3,868, every rate lower except the one
at 100%, as the rules describe), scored with the contest's exact loss. Across the family
(new-cohort profile from `rate^0.5` to `rate^1.5`, flat shares 0.10–0.25, and "new players hold
nothing"), this program's loss is **0.19–0.37 against persistence's 0.88–1.69**; the hypotheses
that reproduce persistence's known practice loss (≈1.51) put this program at ≈0.19–0.24. The one
hypothesis it loses is "no refresh happens at all" (1.08 vs 0.02) — that is the insurance premium,
and the rules state a refresh did straddle the practice window. The final-case machinery
(jump estimation over a multi-day history) was exercised on a synthetic case with known ground
truth (decaying player growth, hazard adoption, rank-shaped cohort profile): loss **0.219 vs
persistence 0.344**, and 0.223 vs 0.287 under a constant-hot-growth stress variant. Output is
byte-for-byte deterministic across runs and a case runs in tens of milliseconds.

## Layout

- `main.py` — the whole forecaster (the only entrypoint; no `main.ts`).
- `LICENSE` — MIT.
- `.makefun/owner.json` — the entrant wallet claim file.

Run it the way the grader does, outside the sandbox:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
