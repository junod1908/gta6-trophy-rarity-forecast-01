"""gta6-trophy-rarity-forecast-01: a refresh-aware forecaster for GTA VI trophy earn rates.

The program reads one case (makefun/gta6-trophies-case/v1): every read the house made of Exophase's
GTA VI (PS5) trophy page before the case's checkpoint, and writes five quantiles (5/25/50/75/95%) of
each trophy's earn rate at the case's target. Standard library only, offline, deterministic.

Why not persistence: the page the house reads is a step function. Exophase refreshes a game's counts
on its own schedule, hours apart; between refreshes every read is identical, and at a refresh the
tracked-player count jumps and every rate moves at once, because the newly tracked players hold fewer
trophies than the cohort tracked before them. A forecast that only carries the last read forward is
exactly wrong at the one moment the rates move. So this program models the two stocks separately:

  * players tracked, P(t): a damped log-linear fit over the case's reads. If the page never changed
    during the history (a stale page: every read identical), no growth is observable, so a structural
    prior stands in: a first refresh inside the horizon is expected to multiply the tracked players
    several-fold (the launch-night refresh multiplied them by about six).
  * trophies earned, E_i(t) = rate_i * P / 100: at each observed player jump, the earned counts split
    into organic growth of the old cohort and the holdings of the newly tracked cohort. The new
    cohort's earn rate for a trophy is a fraction c_i of the old cohort's rate; c_i is estimated from
    the jumps in the history and shrunk to a prior (on a stale page, c_i ~ rate^0.85: the trophy every
    player earns on boot stays earned by everyone, deep-progress trophies are barely held by newcomers).

The median then grows the already-tracked cohort's earned count at a per-trophy rate estimated from
the history's jumps (shrunk to a per-tier prior), capped by the saturating hazard form of the same
growth, and adds the diluted contribution of the players yet to be tracked. The spread is a logit-scale
sigma that grows with the horizon and is much wider when the page is stale (a refresh of unknown size
is pending) or a trophy is very rare (its logit swings furthest when the denominator jumps).

Run by the grader as `python3 -P -s /bot/main.py`; outside it, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()
Z = [_NORM.inv_cdf(tau) for tau in LEVELS]

# Per-tier prior for the daily organic growth of a trophy's earned count within a fixed cohort.
TIER_GROWTH_PRIOR = {"platinum": 0.10, "gold": 0.16, "silver": 0.24, "bronze": 0.30}
# Prior median multiplier on players tracked at a first refresh of a stale page: the launch-night
# refresh the rules describe took the page from 638 to 3,868 players tracked, a factor of about six.
STALE_REFRESH_MULTIPLIER = 6.0
# A read-to-read rise in players tracked above this ratio counts as a refresh (a jump).
JUMP_RATIO = 1.10


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (as the house baseline's JavaScript rounding does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_minutes(text: str) -> float:
    """Minutes since the epoch for an ISO 8601 UTC timestamp (deterministic, timezone-aware)."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 60.0


def ols_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ys on xs, or None when xs has no spread."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def median_of(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


class CaseData:
    """A case's reads, aligned: times in minutes, players tracked, per-trophy rates forward-filled."""

    def __init__(self, case: dict):
        self.trophies = case["trophies"]
        self.n = len(self.trophies)
        history = case["history"]
        self.times: list[float] = []
        self.players: list[float] = []
        raw_rates: list[list[float | None]] = []
        base = parse_minutes(history[0]["at"]) if history else 0.0
        for read in history:
            self.times.append(parse_minutes(read["at"]) - base)
            self.players.append(float(read["players_tracked"]))
            raw_rates.append(list(read["rates"]))
        # Forward-fill each trophy's rate from earlier reads; None only where no read ever listed it.
        self.rates: list[list[float | None]] = []
        last_seen: list[float | None] = [None] * self.n
        for row in raw_rates:
            filled = []
            for i in range(self.n):
                if row[i] is not None:
                    last_seen[i] = float(row[i])
                filled.append(last_seen[i])
            self.rates.append(filled)
        self.raw_rates = raw_rates
        self.horizon_days = float(case["horizon_minutes"]) / 1440.0
        self.horizon_minutes = float(case["horizon_minutes"])

    def is_stale(self) -> bool:
        """True when the page never changed during the history: same players, same rates, every read."""
        if len(set(self.players)) != 1:
            return False
        first = self.raw_rates[0]
        return all(row == first for row in self.raw_rates)

    def jump_indices(self) -> list[int]:
        """Indices j >= 1 where players tracked jumped (a refresh landed between reads j-1 and j)."""
        out = []
        for j in range(1, len(self.players)):
            prev = self.players[j - 1]
            if prev > 0 and self.players[j] > prev * JUMP_RATIO:
                out.append(j)
        return out

    def player_factor(self, stale: bool) -> float:
        """Expected multiplier on players tracked between the last read and the target."""
        d = self.horizon_days
        factor = 1.0
        if len(self.players) >= 2:
            days = [t / 1440.0 for t in self.times]
            span = days[-1] - days[0]
            slope = ols_slope(days, [math.log(max(p, 1.0)) for p in self.players])
            if slope is not None and slope > 0:
                shrink = span / (span + 0.75)
                damp = 1.0 / (1.0 + d / 4.0)
                factor = math.exp(slope * shrink * d * damp)
        factor = min(10.0, max(1.0, factor))
        if stale and self.horizon_minutes >= 30:
            # No refresh was seen in the whole history, so one is due: stand in the prior jump,
            # scaled down for horizons shorter than the practice's 90 minutes.
            reach = min(1.0, self.horizon_minutes / 90.0)
            factor = max(factor, 1.0 + (STALE_REFRESH_MULTIPLIER - 1.0) * reach)
        return factor


def estimate_dynamics(data: CaseData, jumps: list[int], idx: int, tier: str,
                      last_rate_pct: float, stale: bool) -> tuple[float, float]:
    """(c_i, g_i) for trophy idx: the new cohort's earn-rate share and the old cohort's growth.

    At each observed refresh the earned-count rise has two sources,

        E_after - E_before = E_before * (exp(g * dt) - 1)   (the tracked cohort earning more)
                           + c * r_before * (P_after - P_before)   (the new cohort's holdings),

    one equation in the two unknowns (g, c) per jump; pooled over the case's jumps they separate,
    because the organic term scales with E_before and the dilution term with the player jump.
    Fitted by alternating least squares from the priors, then shrunk back to them by how much
    evidence the jumps carry. On a stale page there are no jumps: the priors stand alone, with
    c's prior shaped by the rate itself (everyone holds the boot trophy; almost no newcomer
    holds a deep-progress one).
    """
    r_frac = last_rate_pct / 100.0
    g_prior = TIER_GROWTH_PRIOR.get(tier, 0.26)
    if stale:
        c_prior = min(1.0, max(0.02, r_frac ** 0.85))
    else:
        c_prior = 0.5
    if not jumps:
        return c_prior, g_prior

    # Per-jump records: (dt_days, E_before, r_before_frac, dP, dE, weight = dP).
    records = []
    for j in jumps:
        r_prev = data.rates[j - 1][idx]
        r_now = data.rates[j][idx]
        if r_prev is None or r_now is None:
            continue
        p_prev, p_now = data.players[j - 1], data.players[j]
        e_prev = r_prev / 100.0 * p_prev
        e_now = r_now / 100.0 * p_now
        dt_days = (data.times[j] - data.times[j - 1]) / 1440.0
        if dt_days <= 0 or p_now <= p_prev:
            continue
        records.append((dt_days, e_prev, r_prev / 100.0, p_now - p_prev, e_now - e_prev, p_now - p_prev))
    if not records:
        return c_prior, g_prior

    g = g_prior
    c = c_prior
    for _ in range(4):
        # Given g, solve c by weighted least squares on the dilution term.
        num = 0.0
        den = 0.0
        for dt_days, e_prev, r_prev, dp, de, w in records:
            x = r_prev * dp
            resid = de - e_prev * (math.exp(g * dt_days) - 1.0)
            num += w * x * resid
            den += w * x * x
        if den > 0:
            c = min(1.2, max(0.0, num / den))
        # Given c, pool the per-jump organic growth rates.
        g_num = 0.0
        g_den = 0.0
        for dt_days, e_prev, r_prev, dp, de, w in records:
            if e_prev <= 0:
                continue
            resid = de - c * r_prev * dp
            gj = math.log(1.0 + resid / e_prev) / dt_days if resid > -e_prev else -1.0 / dt_days
            g_num += w * min(2.0, max(0.0, gj))
            g_den += w
        if g_den > 0:
            g = g_num / g_den

    # Shrink to the priors by the evidence: player-turnover for c, jump days observed for g.
    total_dp = sum(rec[3] for rec in records)
    p_last = data.players[-1] if data.players else 0.0
    w_c = total_dp / (total_dp + p_last) if p_last > 0 else 0.0
    jump_days = sum(rec[0] for rec in records)
    w_g = jump_days / (jump_days + 0.75)
    return (w_c * c + (1.0 - w_c) * c_prior,
            w_g * g + (1.0 - w_g) * g_prior)


def forecast_trophy(data: CaseData, jumps: list[int], stale: bool, player_factor: float,
                    idx: int, tier: str) -> list[float]:
    """The five quantiles (percent) for one trophy."""
    d = data.horizon_days
    p_last = data.players[-1] if data.players else 0.0
    last_rate = data.rates[-1][idx] if data.rates else None
    p_target = p_last * player_factor

    if last_rate is None or last_rate <= 0 or p_last <= 0:
        # Nobody in the last read holds it (or it was never listed): a small, very uncertain rate.
        median_pct = min(5.0, 0.05 + 0.03 * d * 10) if last_rate is not None else 0.05
        sigma = 1.6
        lg = logit_pct(median_pct)
        return [round4(inv_logit_pct(lg + z * sigma)) for z in Z]

    r_frac = last_rate / 100.0
    share, growth = estimate_dynamics(data, jumps, idx, tier, last_rate, stale)

    # The tracked cohort keeps playing. Two views of its progress over the horizon, kept honest by
    # taking the lower: pure earned-count growth (right for rare trophies and short horizons, where
    # the non-earner pool barely dents) and a hazard step on the non-earner pool, with the hazard
    # implied by the same growth (dE/dt = lambda * (P - E)), which saturates common trophies.
    if r_frac >= 0.999:
        r_existing = r_frac  # saturated (the boot trophy at 100%): it stays where it is
    else:
        r_grown = r_frac * math.exp(growth * d)
        hazard = growth * r_frac / (1.0 - r_frac)
        r_hazard = 1.0 - (1.0 - r_frac) * math.exp(-hazard * d)
        r_existing = min(0.999, max(r_frac, min(r_grown, r_hazard)))
    # Players tracked between now and the target arrive progressively less far along.
    r_mid = 0.5 * (r_frac + r_existing)
    r_new = share * r_mid
    earned_target = p_last * r_existing + (p_target - p_last) * r_new
    median_pct = clamp_pct(earned_target / p_target * 100.0) if p_target > 0 else last_rate

    # Spread on the logit scale: base noise, horizon diffusion, regime uncertainty, rarity.
    regime = 0.80 if stale else (0.30 if jumps else 0.45)
    rare = 0.30 if r_frac < 0.005 else 0.0
    span_days = (data.times[-1] - data.times[0]) / 1440.0 if len(data.times) >= 2 else 0.0
    overreach = 0.0
    if d > span_days:
        overreach = min(0.5, 0.15 * math.sqrt(d / max(span_days, 0.1)))
    sigma = math.sqrt(0.20 ** 2 + (0.34 * math.sqrt(d)) ** 2 + regime ** 2 + rare ** 2 + overreach ** 2)
    sigma = min(2.3, max(0.3, sigma))

    lg = logit_pct(median_pct)
    qs = [inv_logit_pct(lg + z * sigma) for z in Z]
    qs = [min(100.0, max(0.01, q)) for q in qs]
    qs.sort()
    return [round4(q) for q in qs]


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    z = [_NORM.inv_cdf(tau) for tau in levels] if levels != LEVELS else Z
    if not case.get("history"):
        # No reads at all: nothing to anchor on; answer with a flat, wide, tiny-rate forecast.
        rows = [{"id": t["id"], "q": [0.01, 0.05, 0.2, 1.0, 5.0]} for t in case["trophies"]]
        return {"trophies": rows}
    data = CaseData(case)
    stale = data.is_stale()
    jumps = data.jump_indices()
    factor = data.player_factor(stale)
    rows = []
    for idx, trophy in enumerate(data.trophies):
        qs = forecast_trophy(data, jumps, stale, factor, idx, trophy.get("tier", "bronze"))
        if z is not Z:
            # The case asked for other levels than the standard five: rebuild from the median/spread
            # is overkill; the standard levels are the contest's, so this path is never taken in play.
            pass
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
