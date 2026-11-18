# SupeRR Selector S2 · Round 1 — "Most Reliable Players Under Pressure"
### Master Understanding + Findings + Build Plan (v1, for review before hand-off to coding agent)

> **Status:** v1 draft. Everything in §3–§6 is *measured from your actual files* (prototype code run during analysis). §7 is the build plan. §10 lists the few decisions we should lock together before this goes to Opus/the coding agent.
> **Today:** 13 Nov 2026 · **Deadline:** 18 Nov 2026, 11:59 PM IST (≈ 5 days, from your Unstop screenshot).

---

## 0. TL;DR (read this first)

1. **The task:** rank the **10 best batters** and **10 best bowlers *under pressure*** (Rank 1→10) from IPL + T20I data (2022→2026), and submit a **methodology report** that someone else could reproduce. Rankings must be *data-justified, not reputation*.
2. **There is no "right answer" key.** Judges will score *rigor of your pressure definition, fairness of comparison, evidence, and honesty about uncertainty*. The method is the product; the top-10s are its output.
3. **The 5 things in this dataset that will silently wreck a naive analysis** (all verified):
   - `runs` = **bat runs only**. Wides / byes / leg-byes / no-ball penalty runs are **not stored as numbers** (only as type codes) → exact team totals and chase targets must be *estimated* (±~3 runs).
   - **Legal-ball logic is non-obvious**: wide, `Overstep` (no-ball) **and** `other` are all illegal deliveries.
   - **IPL ≠ T20I** scoring (RPO 9.3 vs 8.4; avg 1st-inn ≈ 185 vs 168). Pooling without adjusting is unfair.
   - **Weak opposition inflates stats**: associate nations (Kuwait, Hong Kong, Scotland…) and "A" teams (Pakistan A, Bangladesh A…) are in the data. A naive leaderboard is *dominated* by these players (shown in §6.6).
   - **Pressure performance is noisy and weakly persistent** (split-half reliability: batters r≈0.17–0.24, bowlers r≈0.05–0.07). A raw "best in pressure" table is mostly luck. → **Shrinkage + consistency + uncertainty-aware ranking are mandatory**, not nice-to-have.
4. **Recommended approach (one line):** *Leverage-Index pressure (from a win-probability model) → context-adjusted run-value per ball → leverage-weighted, opposition/venue-adjusted, empirically-shrunk player ratings → blended with a "level + resilience-vs-own-baseline + match-by-match consistency" score → confidence-aware ranking, stress-tested under alternate pressure definitions.*

---

## 1. Understanding the problem statement

### 1.1 Layman version
Cricket has "easy" balls and "scary" balls. Scoring 4 off the first ball of the match is nice; scoring 4 when you need 12 off 6 balls with 2 wickets left decides the match. Some players shrink in those moments, some grow. Task: use ball-by-ball history to **score every ball by how scary the moment was**, measure **who plays better than normal in scary moments**, and **rank the top 10 batters and top 10 bowlers**.

### 1.2 Intermediate version
Three sub-problems hide inside the one question:
| # | Sub-problem | What it really asks |
|---|---|---|
| A | **Define pressure** | Turn match state (score, wickets, balls left, target, closeness, stage) into a number per delivery. Prove it separates "pressure" from "ordinary". |
| B | **Measure performance fairly** | A 20-off-8 at the death ≠ 20-off-8 in the powerplay. Need a *context-adjusted* value per ball for batters **and** bowlers. |
| C | **Rank reliably** | Small samples, different competitions, different opposition. Need a rating that survives "was this luck?". |

### 1.3 Advanced version
- This is **leverage-conditioned player valuation with shrinkage** — the same family as Leverage Index/WPA (baseball sabermetrics), expected-runs-above-average, and regularised adjusted plus-minus (RAPM).
- The "compared with their usual performance levels" clause demands a **player-specific baseline** (own non-pressure performance) → a *difference/ratio* ("resilience") metric, not just absolute level.
- "Consistently" demands a **dispersion/hit-rate** component at match level, not just a mean.
- "Fair comparisons: sample sizes, eligibility, match contexts, IPL vs T20I" is a rubric bullet → each needs an *explicit, documented* treatment (see §7.5–7.8).
- "Reliability and limitations: how you checked consistency" → we need **split-half reliability, bootstrap rank stability, sensitivity to the pressure definition**. Our prototype already shows why this matters.

### 1.4 What judges will likely reward (inferred from the PS "report must cover" list)
1. Pressure definition: indicators + *why* + proof it discriminates.
2. Metrics, baselines, rating logic (transparent/reproducible).
3. Fair-comparison handling (samples, eligibility, contexts, IPL vs T20I).
4. Evidence: tables/charts supporting the top-10 *and its order*.
5. Reliability + limitations (honest).
Plus: dropdown picks **must exactly match** the ranks in the report; **no duplicates** within a list.

### 1.5 Submission mechanics (from PS + screenshot)
- **Two top-10 lists via dropdowns on Unstop** (Rank 1→10 batters; Rank 1→10 bowlers).
- **Methodology report upload** (PDF / PPT / dashboard acceptable per Unstop page).
- The Unstop page text also says the submission should explain *"top picks, and how each player could strengthen the RR squad"* → include a **short RR-fit section** per player (role, phase, position, what they add). Cheap to add, protects against the page/PS wording mismatch.
- Optional "share your journey" content: not scored; skip unless time.
- ⚠️ **Dropdown ambiguity risk:** the same name can map to different player IDs (see §3.3, e.g. *two* "Rashid Khan"s). We must record **player_id + team** in the report so the pick is unambiguous.

---

## 2. Cricket primer needed to model this (T20)
- 20 overs × 6 legal balls = **120 legal balls/innings**, 10 wickets. Phases: **Powerplay 1–6, Middle 7–15 (I split 7–10 / 11–15), Death 16–20**.
- **Chasing (2nd innings)** has an explicit target → **required run rate (RRR)** vs **current run rate (CRR)**; the cleanest pressure signal.
- **1st innings** has no target; pressure = collapse (wickets down early), stagnation, death-over acceleration.
- **Win probability (WP)** = P(batting team wins | state). **Leverage Index (LI)** = how much the *next ball* can swing WP, relative to average. High LI = "game on the line".
- Wicket credit: bowler gets Bowled/Caught/LBW/Stumped/Hit-wicket; **not** run-outs / retired out / obstructing.
- Bowler is charged bat runs + wides + no-balls; **not** byes/leg-byes.

---

## 3. Data audit (what each file really is)

### 3.1 File inventory
| File | Rows | What it is | Key use |
|---|---|---|---|
| `ball_by_ball.csv` | 200,999 (33 cols) | One row per **delivery** (incl. wides/no-balls). 867 matches. | Core of everything |
| `matches.csv` | 883 | One row per match: date, stage, venue_id/country, teams, toss, **result text**, `win_team_id`, DLS fields, `knock_out`. | Outcome labels, venue, stage, DLS flags |
| `competitions.csv` | 141 | comp_id, name, **match_format (T20I=134, T20=7)**, season | IPL vs T20I vs "A"-team comps |
| `teams.csv` | 353 | comp_id × team_id × team_name | Team tiering |
| `competition_team_player_mapping.csv` | 6,413 | Squad lists: player_id, player_name, team, comp | **Canonical player names/IDs**, team labels |
| `playing_eleven.csv` | 28,381 (880 matches) | Squads/XI per match: batting_order, `is_played`, captain, wk, **`is_impact`/`is_impact_off`** | Matches-played, impact-sub context |
| PDFs (×2, identical) | – | Problem statement | – |
| PNG | – | Unstop stage page (dates, submit flow) | – |

### 3.2 `ball_by_ball.csv` column semantics (verified, not assumed)
- **Keys/ordering:** `match_id, innings, overid (1–20), over_ball_num (1–13), innings_ball_num` (counts *every* delivery incl. wides/no-balls; restarts each innings).
- **Players:** `striker_id/name, non_striker_id/name, bowler_id/name, batting_order (1–13), batting_type (RHB/LHB), bowling_type (12 categories), pace_or_spin`.
- **`runs`** = **runs off the bat only** (0–6). Wide rows always have `runs=0`; leg-bye/bye rows have `runs=0`.
- **`extras_type`** ∈ {NaN, wide 7,714, legbyes 2,748, Overstep 576 (=no-ball), byes 529, other 264, Penalty 1}.
- **`extras1_runs` is a CODE, not runs:** `6`=no extra, `2`=wide, `3`=legbye, `4`=bye, `1`=Overstep, `5`=other, `7`=penalty. **Do not sum it as runs.**
- `extras_2_runs` ∈ {0,1,4}; only ~36 non-zero rows (negligible).
- **Wickets:** `is_wicket` (10,881 rows), `dismissal_type` {Caught 7,557; Bowled 1,775; LBW 709; RunOut 600; Stump 214; Hit Wicket 17; Retire Out 6; Obstructing 3 (two spellings)}, `dismissed_player_id` (equals striker 97.5%; **271 run-outs dismiss the non-striker**; 1 NaN = a retire-out).
- **Likely useless/low-trust:** `bowl_speed` (≈50% zeros/negatives, max 1456), `delivery_ist_time` (sample shows date-only/midnight), `stroke`, `delivery_style`, `over_or_around`, `on_or_off` (some NaN). Not needed.
- **Innings:** 1 (105,727 rows), 2 (95,234), **3 & 4 = super overs (38 rows, 6 matches) → drop**.

### 3.3 Critical data gotchas (each one changes results)
1. **Extras run-values are missing.** Reconciling innings totals against `matches.result` ("won by X runs") on 396 run-margin matches: bat-runs-only is wrong by MAE 5.5 runs; adding estimated extras brings MAE to **≈3 runs** (can't be driven to zero). Least-squares average extras values: **wide ≈ 1.0, no-ball ≈ 1.0 (+ bat runs already recorded), leg-bye ≈ 1.5, bye ≈ 1.5, other ≈ 0.4, penalty 5**. → team score/target = **estimate**, so pressure is a *soft* score; treat ±3 runs as noise (documented limitation).
2. **Legal-ball rule (verified 99.99% of full overs = 6 legal):** illegal = `wide`, `Overstep`, `other`. (`Penalty` ≈ no ball consumed.) `balls_left = 120 − legal balls before this delivery`. Batter "balls faced" excludes wides.
3. **Duplicate display names with different IDs (7 names / 14 IDs):** Abdul Samad (Pak vs IPL), Bashir Ahmad, Basil Hameed, Haider Ali (Pak vs UAE), Mohammad Ali, Muhammad Farooq, **Rashid Khan (Afghanistan/GT id 4722 vs Nepal id 20172)**. **Always key on `player_id`.** Names within ball data are consistent per ID; **all** ball-data IDs exist in the mapping.
4. **Coverage:** 883 matches listed, **867 have ball data**; the 16 missing are all "Match Abandoned". 21 "No Result", 5 ties (super over; one tie has NaN winner). **34 DLS/revised matches** (4,433 balls) → exclude from win-probability training & from pressure scoring (or handle separately).
5. **Competition mix:** IPL 2022–2026 (371 matches, ~94.7k balls) + **2 "A"-team T20 tournaments** (Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025; 30 matches; Pakistan A, India A, Bangladesh A, Sri Lanka A, Afghanistan A, HK, Oman, UAE) + **134 T20I competitions** (482 matches, 23 teams, incl. associates). **No ICC World Cup competition is present** (searched names). Only **31 knock-out-stage matches** (Final 12, Semi 4, Qualifier 10, Eliminator 5) → stage can be at most a small modifier.
6. **Season labels** are messy ("2022/23"); use `match_date`. Dates run **2022-01-22 → 2026-11-01** (IPL 2026 complete; T20Is through Oct/Nov 2026 incl. India in England/NZ, etc.).
7. **Venue** only has `venue_id` + country (95 venues; only 33 with ≥8 matches) → venue adjustment must be shrunk/hierarchical.
8. **Result strings inconsistent** ("Won by"/"won by", "Runs"/"runs") → regex case-insensitively; prefer `win_team_id`.
9. `playing_eleven`: ≈22 `is_played=True` per match (others = bench/squad listing); 558 `is_impact` rows → impact subs exist (IPL 2023+); `batting_order` in ball data can reach 12–13 for them.
10. **Exposure (ball-by-ball, innings 1–2, non-DLS):** batters with ≥200 balls faced: **212**; ≥100: 319. Bowlers with ≥200 legal balls: **223**; ≥100: 330. Players in both IPL & T20I data: **199** (IPL 380 players, T20I 701).

### 3.4 Baseline scoring environments (verified)
| | RPO inn1 | RPO inn2 | Wkts/over | Dot% | 6s/over | Avg 1st-inn (est.) |
|---|---|---|---|---|---|---|
| **IPL** | 9.32 | 9.20 | 0.32 | 37.6–38.6 | 0.45 | **≈184.6** |
| **T20I** | 8.55 | 8.32 | 0.34–0.36 | 39.8–41.5 | 0.37 | **≈168.4** |
Death-over RPO: IPL 10.74 vs T20I 9.80; PP: 9.05 vs 8.20. **→ every expected-value table must be built per competition group, or players are unfairly rewarded for playing in the IPL (or T20I).** 54 of 482 T20Is involve an associate side.
Overall chase win rate = 0.50 (818 matches) → no structural chase/defend bias.

---

## 4. Method design — decisions and reasoning

### 4.1 Pipeline overview
```
raw CSVs
  → clean/state reconstruction (legal balls, est. extras, score, wkts, balls_left, target)
  → win-probability (WP) models (chase + 1st inn) → Leverage Index (LI) per ball = PRESSURE
  → run-expectancy V(balls_left, wkts) per comp-group → ball Run Value (RV) for batter / bowler
  → adjustment (comp group, opposition quality, venue, tier weights)
  → player aggregation: leverage-weighted mean, empirical-Bayes shrinkage, uncertainty
  → components: Level (pressure perf) + Resilience (vs own baseline) + Consistency (match hit-rate)
  → composite → confidence-aware ranking → robustness (alt. definitions, bootstrap) → Top-10 + report
```

### 4.2 Pressure definition (primary): **Leverage Index from a win-probability model**
**Why:** One principled number that automatically combines required rate, wickets in hand, balls left, and closeness — the exact list the PS suggests — and is comparable for both innings and for batting *and* bowling (same ball is pressure for both sides). It is also *data-driven* rather than arbitrary thresholds.

**Construction (prototype-verified):**
- **Chase WP model:** HistGradientBoosting on `balls_left, wickets_in_hand, runs_needed, RRR, CRR, RRR−CRR, score, ipl flag`; grouped (by match) 5-fold CV → **AUC 0.893, log-loss 0.409 vs 0.692 baseline, well-calibrated** (AUC by balls-left bucket: 0.80 early → 0.956 last 30 balls).
- **1st-innings WP model:** same family on `score, wickets, balls_left, CRR`; **AUC only 0.674** (0.57 early → 0.73 late). *First innings is inherently less predictive; treat its pressure as lower-confidence.*
- **LI per delivery** = Σ over next-ball outcomes (0,1,2,3,4,6,W,wide,no-ball; probabilities from league outcome frequencies by innings × phase × wickets bucket) of `P(outcome) × |WP(next state) − WP(current)|`, divided by the mean → `li` (mean = 1).
- Observed: chase `li` mean 1.10 (std 1.16, max 18.6); inn1 mean 0.91; overall percentiles: p50 0.79, p75 1.33, p90 2.04, p95 2.62, p99 4.31. **LI≥1.75 ≈ 14.5% of balls; LI≥2 ≈ 5.7% (inn1) / 15.8% (inn2).**
- In the chase, LI peaks when WP is 0.35–0.65 and rises sharply in the last 12–24 balls (LI 1.76 last 12 balls) — exactly "the game is on the line".

**Proof that it is "real" pressure (the "bite test" — include in report):** as LI rises, batting outcomes deteriorate:
| Chase innings | LI <0.75 | 1.25–2 | 2–3 | **3+** |
|---|---|---|---|---|
| Wicket / 100 balls | 5.81 | 5.23 | 5.95 | **7.63** |
| Bowler-credited wkt / 100 | 5.58 | 4.95 | 5.54 | **7.14** |
| Dot % | 37.3 | 39.0 | 39.8 | 36.0 (batters forced to attack: SR 151.8) |
| 1st innings: Dot % | 34.6 | 39.2 | 41.2 | **43.7** |
→ *Raw strike rate rises under chase pressure (forced aggression) — so **SR/average are invalid pressure metrics**; context-adjusted value is required.* This is a compelling report figure.

**Known weakness (must be disclosed and handled):** 1st-innings LI is dominated by an early-innings artefact (mean LI 1.33 in over 1, 0.7–0.75 mid-innings, lowest overs 8–15) because the 1st-inn WP model is weak/steep early. Chase LI in over 1 (1.80) is also partly "model sensitivity", not true pressure.
**Handling (to implement + sensitivity-test):**
- (i) **Reliability-weight LI by phase**: `li_eff = li × ρ(innings, phase)` with ρ = relative informativeness of the WP model (≈ (AUC_phase−0.5)/(AUC_chase_late−0.5)); or
- (ii) simply **apply a minimum-decision-window rule** (pressure counts only from ball ≥ 25 in inn1 unless ≥3 wickets down; no early-over chase LI discount beyond ρ).
- (iii) Whatever is chosen, run the **alternate pressure definitions (§4.3)** and report ranking stability.

### 4.3 Alternate / robustness pressure definitions (pre-register these; do NOT cherry-pick after seeing rankings)
| ID | Definition (transparent, rule-based) | Why include |
|---|---|---|
| **P-LI** (primary) | `li_eff ≥ T` (try T∈{1.25,1.5,1.75,2.0}) and also **continuous weighting** `w=clip(li,0.25,4)^γ` (γ∈{1,1.5}) | Principled, uses all data |
| **P-Chase** | Chase, `balls_left ≤ 48`, `0.2 ≤ WP ≤ 0.8` (≈4.2% of balls; 71% overlap with P-LI≥1.75) | "Game on line" in plain English |
| **P-Squeeze** | Chase with `RRR − CRR ≥ 1.5` or `RRR ≥ 10` and `balls_left ≤ 60` | Pure required-rate pressure |
| **P-Collapse** | Any innings: `wickets_before ≥ 3` before ball 36, or ≥ 5 before ball 72 (excess-wicket state) | Rebuild pressure (1st-inn fix) |
| **P-Death** | Overs 16–20 with `|WP−0.5|≤0.25` (chase) / innings-1 death | Death-overs specialists |
Final ranking = **primary P-LI**, but a player only enters the Top-10 if he is **robust (top-15 rank in ≥3/5 definitions or Top-10 inclusion probability high)**.

### 4.4 Performance metric: context-adjusted **Run Value (RV) per ball**
Why runs-based: split-half reliability of run-component (batters 0.36–0.44, bowlers 0.14–0.28) ≫ wicket component or combined, and it is context-adjusted by phase/wickets (not raw SR/economy).
- **Run-expectancy table V(b, w)** = expected *remaining* team runs with `b` balls left, `w` wickets lost. Built by **dynamic programming** from empirical per-over × wicket tables of (a) mean runs/ball and (b) P(wicket/ball), smoothed across neighbouring overs/wicket buckets (a naive smoothing or GBM version gave nonsense at the death — DP is consistent: **team RV mean = 0.019 (inn1) / −0.010 (inn2)** ✓). V(120,0)≈174; V(60,0)≈107; V(12,0)≈25; V(1,0)≈1.7.
  **→ Build V separately for IPL and T20I (prototype used pooled; must fix).**
- **Per-ball components:** `time_term = V(b−legal, w) − V(b, w)`; `wk_cost = V(b', w+1) − V(b', w)` (negative).
  - **Batter RV** (striker, non-wide) = `bat_runs + time_term` **+** `wk_cost` assigned to the **dismissed player** (so a non-striker run-out hurts the non-striker, not the striker). Centre by league mean (extras offset ≈ +0.3/ball).
  - **Bowler RV** = `−(bat_runs + 1·[wide|no-ball] + time_term) − wk_cost·[bowler-credited dismissal]`. Byes/leg-byes not charged.
- **Why RV and not strike rate/average/economy:** RV prices wickets, phase, and scoreboard state in one currency and is ≈ zero-mean by construction.
- **Secondary metric (report only / robustness):** **WPA** = `WP_after − WP_before` credited to striker/bowler (last ball of inn1: compare with chase-start WP from model; final ball of chase: actual result). Good for storytelling ("matches won"), too noisy to rank on alone.

### 4.5 Baseline and "compared with usual performance"
- **Context-centre** each ball against league mean **within (comp-group × innings × LI-bucket)** — *essential*: prototype showed nearly every player has negative "non-pressure" RV when pressure-ball RV is centred globally, because pressure balls have systematically higher RV (batters forced to attack). Compare like with like.
- **Player baseline** = the player's own leverage-low RV (e.g., LI < 0.75), adjusted/shrunk the same way.
- **Resilience (Δ)** = `Rating_pressure − Rating_baseline` (both shrunk), with its own uncertainty. Use as a **component**, *not* the sole criterion (a great player who is equally great under pressure should still rank high — PS says "best-performing under pressure").

### 4.6 Rating model (recommended): leverage-weighted, adjusted, **empirically-Bayes-shrunk**
Per player `i` (batter or bowler separately):
1. **Adjusted ball values** `v_adj = RV_centred − opponent_effect − venue_effect` where opponent/venue effects come from a **ridge regression on all balls** (batter + bowler + venue + comp-group one-hots, joint → RAPM-style) *or* a simpler hierarchical team-strength rating. (Prevents associate/“A”-team feasting; §4.8.)
2. **Pressure-weighted mean** `m_i = Σ w·v_adj / Σ w` with `w = tier_weight × clip(li_eff,0.25,4)^γ`. Effective sample `n_eff,i = (Σw)²/Σw²` (or Σw).
3. **Empirical-Bayes shrinkage** to 0 (league avg): `post_i = m_i × τ² /(τ² + σ²/n_eff,i)`; prototype: τ≈0.26–0.29 RV/ball, **median shrink factor ≈ 0.75**; rank on a **conservative score**: posterior mean − k·posterior SD (k≈1.0–1.28, i.e. 10th–16th percentile "lower credible bound") *and* publish the plain posterior mean ranking for comparison.
4. **Consistency component** `C_i` (match level): for each match where the player had ≥ N pressure-weighted balls (batters ≥ 6 balls with `li≥1`; bowlers ≥ 6 legal balls with `li≥1`), compute match-level pressure RV vs. expectation; `C_i` = EB-shrunk **hit-rate** (% matches ≥ 0) blended with the **20th-percentile** match outcome (downside). Also report % of *seasons/competitions* positive.
5. **Composite** (defaults, to be tuned only via pre-declared sensitivity grid):
   `Score = 0.60·z(Level_post) + 0.20·z(Resilience_post) + 0.20·z(Consistency)`; evaluate weights {0.7/0.15/0.15, 0.5/0.25/0.25, 0.8/0.1/0.1}.
6. **Do not** pick the top by raw SR/average; those appear only as supporting columns.

### 4.7 Eligibility (defaults; verify counts and stress-test)
- **Batters:** ≥ 250 balls faced (tier-1 matches), ≥ 40 high-pressure balls (`li≥1.25`), ≥ 15 matches; **plus** active in 2025–26 (auction relevance) — run both "all-years" and "recent-weighted" (half-life ≈ 2 seasons) and report the difference.
- **Bowlers:** ≥ 300 legal balls, ≥ 60 high-pressure balls, ≥ 15 matches, same recency check.
- Expect ~100–150 eligible per side (prototype: neff≥100 → 361 batters/366 bowlers without tier filtering; with tier + stricter minimums it will shrink — report the exact number).
- Players with tiny samples can still be listed in an appendix as "watchlist" (never ranked).

### 4.8 Fair comparison: IPL vs T20I, opposition, venue, role
- **Competition-group baselines:** separate V tables, WP "ipl" feature, and LI-bucket centring for **IPL / T20I** (and drop "A" tournaments from rankings but optionally use for baselines).
- **Tiering (weights, pre-declared):** T20I/IPL where both sides are full-member/IPL-level → **1.0**; T20I involving an associate side → **0.5** (and still opposition-adjusted); **"A"-team competitions (comp_ids 2670, 3289) → 0** for ranking (report effect of including them). Associates list used: UAE, Netherlands, Hong Kong, Scotland, Canada, USA, Singapore, Kuwait, Nepal, Oman, Japan (Ireland/Zimbabwe/Afghanistan treated as full members—document).
- **Opposition strength:** joint ridge (batter/bowler/team/venue) or team Elo from match results; the adjusted value subtracts the *faced bowler/team* effect from batters (and the faced batter/team effect from bowlers).
- **Venue:** hierarchical shrinkage on venue effect (95 venues; only 33 ≥ 8 matches) — shrink small-n venues toward country/comp-group mean.
- **Role/context:** report batting-order and phase splits for each Top-10 player (openers face more PP pressure; finishers face death). Bowlers: pace/spin share, phase mix. Use as *explanatory columns*, not as ranking inputs — but check the ranking isn't just "who bowls the death" (V already conditions on balls-left/wickets).
- **Impact-player / DLS / super-over / no-result:** drop DLS (34 matches) and innings 3–4; keep otherwise.
- **Data cutoff note:** WP/V are *not* recomputed leakage-free per season; the model is descriptive of 2022–26, which is acceptable for retrospective ranking (state this).

### 4.9 Reliability, validation and what we claim
**Prototype reliability result (important, honest, report it):** random split of matches into halves; correlation of the same player's pressure RV/ball across halves (players with ≥25–40 pressure balls/half; n≈90–130):
| | Pressure→Pressure | Pressure→Non-pressure | Non-pressure→Non-pressure |
|---|---|---|---|
| Batters | 0.17–0.24 | 0.22–0.29 | 0.27–0.33 |
| Bowlers | **0.05–0.07** | 0.16–0.17 | 0.24–0.29 |
*Reading:* pressure performance mostly reflects **general skill + noise**; pure "clutch" signal is small, esp. for bowlers. Component view: run-component more reliable than wicket-component; **hard high thresholds (LI≥2–2.5) hurt reliability; LI≥1.0–1.25 (or continuous weights) is the sweet spot.** → This justifies **continuous leverage weighting + shrinkage + consistency**, and is itself a *credible limitation section*.

**Mandatory validation outputs:**
1. WP model: AUC, log-loss, calibration plot (by innings).
2. Bite test table/plot (§4.2).
3. Split-half reliability of **final composite** (target: beat raw 0.2/0.06) — repeat 200× random splits + **odd/even seasons** split.
4. **Cluster bootstrap by match (≥500 reps)** → each player's **P(Top-10)** and rank interval; final list = highest posterior-mean *with* high P(Top-10).
5. **Sensitivity grid:** pressure def (P-LI thresholds + P-Chase/Squeeze/Collapse/Death) × γ × tier-weights × composite weights × recency. Report Kendall τ / overlap with the final Top-10.
6. **Leave-one-competition-out** (e.g., drop IPL 2026; drop T20I) to show the list isn't driven by one tournament.
7. **Sanity audit:** every Top-10 player: ball counts, #matches, #pressure innings, IPL/T20I split, opposition-tier split, raw SR/Econ for context, *and a plain-English reason they ranked there.*

---

## 5. Prototype findings so far (numbers you can quote)
- WP chase AUC **0.893**; inn1 AUC **0.674**.
- Pressure ≈ top 14.5% of balls at `li≥1.75`; 10.3% avg of a match's balls are `li≥2`.
- Bite test: wicket rate +31% (5.81→7.63/100) at LI 3+ in chases; dot% +9 pts in 1st-inn at LI 3+.
- Player exposure at LI≥1.75: 136 batters with ≥60 pressure balls; **48 with ≥150**. Per-ball RV SD ≈ 1.7 → SE ≈ 0.14 at 150 balls ≈ the size of the "top vs average" gap → **ranking noise is large**.
- Reliability: see §4.9.
- Extras: see §3.3.1. Scoring environment: §3.4.

### 5.1 Directional (NOT submittable) preview
A prototype (leverage-weighted, EB-shrunk, centred by comp×innings×LI-bucket, **no opposition/tier filter yet**) surfaced names such as **Vaibhav Sooryavanshi, Priyansh Arya, Tim David, Abhishek Sharma, Heinrich Klaasen, Reeza Hendricks, Ashutosh Sharma, Phil Salt, Suryakumar Yadav, Prabhsimran Singh, Mitchell Marsh, Nicholas Pooran** among batters, and **Josh Hazlewood, Sunil Narine, Rashid Khan, Jacob Duffy** among bowlers — *but the same lists were polluted by Kuwait's Edson Silva, Hong Kong's Ehsan Khan, Scotland's Bradley Currie, Pakistan A / Bangladesh A / Afghanistan A players, etc.* That contamination is exactly why **§4.7–4.8 tiering + opposition adjustment is Step 1 priority**. Treat all names above as hypotheses only.

---

## 6. Open questions the data can't answer (for transparency)
1. True extras run-values (hence exact targets).
2. Players' current/auction status for IPL 2027 (data only contains squads through 2026) → we rank on performance; add a **non-ranking RR-fit flag** (overseas/Indian inferred from national team; role; RR-owned in 2026 per mapping, e.g., Sooryavanshi/RR players will appear — consider noting "already in RR squad" without changing the rank).
3. Pitch/dew/toss effects beyond `venue_id` and `toss_opted`.
4. Bowler quality of *individual* opposition in small samples (handled via ridge).

---

## 7. Build plan for the coding agent (step-by-step)

### 7.0 Environment
Python 3.12; `pandas, numpy, scikit-learn (HistGradientBoosting*, Ridge), scipy, matplotlib/seaborn`; (no lightgbm/statsmodels installed — don't assume). Files at `/mnt/user-data/uploads/*.csv`. Write outputs to `/mnt/user-data/outputs/`. Fix `random_state` everywhere; log all parameters to `config.yaml`.

### 7.1 Repo layout
```
config.yaml                 # all thresholds/weights/tier weights/seeds
src/01_load_clean.py        # joins, flags, legal ball, extras estimate, state vars
src/02_wp_leverage.py       # WP models (chase, inn1), LI per ball, calibration plots
src/03_run_value.py         # DP run expectancy per comp-group, RV per ball (bat, bowl)
src/04_adjust.py            # tiers, ridge opposition+venue effects, centring
src/05_player_ratings.py    # weighted means, EB shrink, resilience, consistency, composite
src/06_validation.py        # reliability, bootstrap, sensitivity grid, LOCO
src/07_outputs.py           # final tables, charts, rankings CSV, report assets
tests/test_*.py             # unit tests (see 7.9)
report/                     # report builder (PDF) + figures
```

### 7.2 Step 1 — Load & clean (`01_load_clean.py`)
- Read all CSVs; merge `matches`+`competitions`; derive `is_ipl`, `comp_group` ∈ {IPL, T20I, A_TEAM}, `year` from `match_date`.
- Drop innings ≥3. Flag `dls` (revised fields non-null) and exclude from WP training/ratings.
- `extras_type` → `ex_runs` map: none 0, wide 1, Overstep 1, legbyes 1.5, byes 1.5, other 0.4, Penalty 5 (**store as parameters; document regression source**). `tot_runs = runs + ex_runs`; `bowler_runs = runs + 1·[wide|Overstep]`.
- `legal = ~extras_type.isin(['wide','Overstep','other'])`.
- Sort by (`match_id, innings, innings_ball_num`); `legal_before`, `balls_left = 120 − legal_before`, `runs_before`, `wk_before` (cumulative `is_wicket`), `wk_in_hand`, `inn1_total`, `target = inn1_total + 1`.
- Outcome label `bat_won = (win_team_id == batting_team_id)`; NaN for no-result/abandoned. For ties, use `win_team_id` (super-over winner) **or exclude** — decide & document.
- **Assert** ≥99.9% of non-final overs have exactly 6 legal balls; ID→name 1:1.
- Team tier table: associates + A-teams (see §4.8).
**Acceptance:** counts match §3 (200,961 rows after dropping inn 3–4; 4,433 DLS rows; 867 matches).

### 7.3 Step 2 — Win probability & Leverage (`02_wp_leverage.py`)
- Train **two** `HistGradientBoostingClassifier` (params in prototype: depth 4, lr 0.05, 250 iters, min_leaf 200, l2 1.0) on non-DLS, non-NR rows; **GroupKFold by match** for OOF predictions used in all evaluation; final fit on all for scoring.
- Features per §4.2. Report AUC/log-loss/calibration; by balls-left bucket.
- Compute `wp_before`; hypothetical next-state WP for outcomes {0,1,2,3,4,6,W,wd,nb}; outcome probabilities from league frequencies by (innings × phase{PP,M1,M2,D} × wicket-bucket{0–1,2–3,4+}); `lev = Σ p·|ΔWP|`; `li = lev/mean(lev)`.
- Apply first-innings/phase reliability weighting (§4.2 handling) → `li_eff`. Save per-ball parquet/CSV (`ball_state.parquet`).
- Also compute `wpa` per ball (secondary).
**Acceptance:** AUC chase ≥ 0.88; bite-test table reproduces §4.2 within tolerance; plot LI by over & WP.

### 7.4 Step 3 — Run value (`03_run_value.py`)
- **DP run-expectancy per `comp_group` (IPL, T20I)**; `R[over,wk]`, `P[over,wk]` smoothed with neighbours + fallbacks for sparse (high-wicket early) cells (**prototype hit NaN here — must fill**); `V[b,w] = R + P·V[b−1,w+1] + (1−P)·V[b−1,w]`; `V[0,·]=0`, `V[·,10]=0`.
- `time_term`, `wk_cost`; `rv_bat` (striker, non-wide), `wk_cost → dismissed_player`, `rv_bowl` per §4.4.
- **Check:** mean team RV ≈ 0 per innings & comp group (|mean| < 0.05).

### 7.4b Step 3b — Pressure flags
Compute all definitions in §4.3 as boolean/continuous columns on the ball table.

### 7.5 Step 4 — Adjustments (`04_adjust.py`)
- Tier weights (config). Centre RV within (comp_group × innings × LI-bucket(8 bins)).
- **Ridge model(s)** (sparse one-hot; α chosen by grouped CV) on *all non-DLS balls*: `v ~ batter + bowler + venue + comp_group (+ batting_order bucket optional)` → extract bowler effect, venue effect; analogous model for bowler perspective. Subtract opponent+venue effects → `v_adj`.
- Report the biggest/smallest bowler/venue/team effects as a sanity table (associate bowlers should show negative "difficulty").

### 7.6 Step 5 — Ratings (`05_player_ratings.py`)
- For each pressure definition: aggregate per player (batters via striker rows + dismissed-player wicket cost; bowlers via bowler rows).
- Compute: `n_balls, n_pressure_balls, n_eff, n_matches, n_pressure_innings, IPL/T20I split`, weighted mean `m`, **EB shrink** (τ² by method-of-moments, per role), posterior mean/SD/LCB.
- Baseline rating (LI<0.75) → **Resilience** = pressure − baseline (shrunk, with SD).
- Match-level consistency (§4.6.4).
- Composite z-scores → `score`, `rank_posterior`, `rank_conservative`.
- Eligibility filter (config) → ranked table; watchlist table.
- **Name safety:** keep `player_id`; produce display name + `(team(s), country)` and flag duplicate display names.

### 7.7 Step 6 — Validation (`06_validation.py`)
Implement every item in §4.9 (reliability on composite; cluster bootstrap ≥ 500; sensitivity grid; LOCO; odd/even seasons). Output `stability.csv` with `P(top10)`, rank CI, overlap across scenarios.

### 7.8 Step 7 — Final selection logic (decide *before* looking at names)
1. Rank by composite posterior (primary definition).
2. Require: eligibility ✓, **P(Top-10) ≥ ~0.5** (or top-15 in ≥ 3/5 alt definitions), non-A-team/associate-driven (≥ X% of pressure balls vs tier-1; config).
3. Ties/near-ties (|Δscore| < bootstrap SE): break by consistency C, then sample size.
4. **Hard checks:** 10 distinct `player_id` per list; each ID resolves to **one** Unstop dropdown name (note ambiguities).
5. Export `rankings_final.csv` (rank, player_id, name, team, role, score, CI, P(top10), key stats) for both lists + `dropdown_picks.txt`.

### 7.9 Tests (must pass)
- Legal-ball: ≥99.9% full overs = 6 legal. ID↔name 1:1.
- Σ ball RV over an innings ≈ (final total − V(120,0)) (± tolerance).
- Team RV mean ≈ 0 by group.
- WP monotonicity spot-checks (more runs needed → lower WP; more wickets in hand → higher WP).
- No duplicate IDs in either top-10; no A-team-only players; eligibility respected.
- Re-run with same seed → identical rankings.

### 7.10 Report (PDF, ~10–14 pages + 1-page exec summary) — suggested outline
1. Executive summary (method in 5 lines + two top-10 tables with one-line reasons)
2. Data & assumptions (incl. extras estimation, legal balls, exclusions: DLS, super overs, A-teams, associate weighting)
3. **Pressure definition** (LI; why; construction; bite-test figure; pressure vs ordinary distribution; alt definitions)
4. **Performance measurement** (RV, run-expectancy, batter/bowler formulas, baselines, resilience, consistency, composite)
5. **Fair comparisons** (sample sizes/eligibility, EB shrinkage, IPL vs T20I treatment, opposition/venue adjustment, phase/role context)
6. **Results: Top-10 batters & Top-10 bowlers** — table (ID, team, balls, pressure balls, matches, rating ±CI, resilience, consistency, P(top10)); rank-order justification; at least: rating-vs-uncertainty "forest" plot, pressure-vs-baseline scatter, per-match consistency strips, phase mix.
7. **Reliability & limitations** (split-half, bootstrap, sensitivity, LOCO, extras uncertainty, weak 1st-inn WP, no World Cups/knockouts sample small, no auction-status data)
8. **RR fit** (2–3 lines per player: phase/role, what he adds, overseas/Indian slot, risk)
9. Appendix: parameters, formulas, reproducibility notes, watchlist.
Charts: WP calibration; LI by over/innings; bite test; reliability bars; forest plots; sensitivity heatmap (overlap of Top-10 across scenarios).

### 7.11 Suggested timeline (to 18 Nov 11:59 PM IST)
| Day | Output |
|---|---|
| 13–14 Nov | Steps 1–4 incl. tiering/opposition adjustment; bite-test + calibration figures |
| 15 Nov | Step 5 ratings + first top-10s; sanity-audit each name |
| 16 Nov | Step 6 validation/sensitivity; lock selection logic |
| 17 Nov | Report build, charts, RR-fit paragraphs, dropdown cross-check |
| 18 Nov | Buffer; submit **well before** 11:59 PM IST (upload + 20 dropdown picks) |

---

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Noisy ranks (reliability r≈0.1–0.2) | EB shrinkage, continuous weights, LCB ranking, P(Top-10), explicit limitation text |
| Weak 1st-innings WP | Phase reliability weighting + P-Collapse alt def + disclose |
| Associate/"A"-team inflation | Tier weights + ridge opponent effects + eligibility |
| IPL/T20I mismatch | Separate V/centring per comp group; show IPL/T20I splits |
| Extras unknown → target error ±3 | Treat pressure as soft; check sensitivity with ex_runs ±; disclose |
| Over-engineering/time | Freeze scope after Step 5; ridge is optional if time-critical — fall back to team-tier weights + EB |
| Dropdown name mismatch | Keep IDs; verify each pick against Unstop list; note duplicates (Rashid Khan etc.) |
| Looking like "reputation picks" | Every pick has a numeric audit trail; no manual overrides — if any override, document it |

---

## 9. Glossary
**RRR/CRR** required/current run rate · **WP** win probability · **LI** leverage index (expected |ΔWP| of next ball ÷ average) · **WPA** win probability added · **RV** run value (runs + change in expected remaining runs, incl. wicket cost) · **V(b,w)** run-expectancy · **EB** empirical Bayes · **LCB** lower credible bound · **RAPM** regularised adjusted plus-minus · **DLS** rain-revised target · **Tier-1** full-member/IPL-level matches.

---

## 10. Decisions to lock with you before the coding hand-off
1. **Tier weights:** associate-involving T20Is = 0.5 vs exclude entirely? ("A"-team comps = excluded.)
2. **Recency:** pure 2022–26 pool, or half-life weighting (auction relevance)? (Proposal: report both, rank on recency-weighted.)
3. **Composite weights** 0.6 / 0.2 / 0.2 (level / resilience / consistency) — OK, or heavier on pure level?
4. **RR-owned players** (e.g., current RR squad) — rank by performance but flag, or filter them out since the Unstop page says "players RR should back at auction"? (Proposal: keep ranked, flag.)
5. **Report format:** PDF (default) vs PDF + small interactive dashboard (extra effort; optional).
6. Do you want the **1st-innings pressure** handled via reliability-weighted LI (proposal) or via simple rule-based Collapse/Death definitions only?

---

## 11. Hand-off prompt for the coding agent (paste with this file + the CSVs)
> You are an expert sports data scientist. Implement the pipeline in `PLAN_SupeRR_Selector_R1_Pressure_Players.md` §7 end-to-end on the uploaded CSVs. Respect all data gotchas in §3.3 (bat-runs-only `runs`, extras are codes, illegal-ball rule, duplicate names → use `player_id`, drop innings 3–4 and DLS matches, separate IPL/T20I baselines, tier/opposition adjustment). Use the locked decisions in §10. Produce: (1) `rankings_final.csv` for top-10 batters and top-10 bowlers with CIs and P(Top-10), (2) all validation artefacts in §4.9, (3) a PDF methodology report following §7.10 with figures, (4) a log of parameters and a reproducibility README. Do not hand-pick players; every ranking must trace to computed scores. Report any deviation from this plan and any failed test.
