"""
03_run_value.py - Dynamic Programming Run Expectancy Table V(b, w),
Run Value (RV) for Batters and Bowlers, and Alternate Pressure Definitions.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §4.4, §7.4, §7.4b.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dp_table(df, comp_group="IPL", M=40.0):
    """
    Builds the dynamic programming run-expectancy matrix V[b, w]
    where b = balls remaining (0..120), w = wickets lost (0..10).
    Boundary conditions: V[0, :] = 0, V[:, 10] = 0.
    Includes empirical extra_rate per legal ball to ensure exact zero-mean team RV balance.
    """
    sub = df[(df["comp_group"] == comp_group) & (~df["is_dls"])].copy()
    if len(sub) == 0:
        sub = df[~df["is_dls"]].copy()

    legal_sub = sub[sub["legal"] == 1].copy()
    n_legal = len(legal_sub)
    # Extra runs from illegal balls (wides/no-balls) per legal ball opportunity
    extra_runs_total = sub[sub["legal"] == 0]["tot_runs"].sum()
    extra_rate = extra_runs_total / max(n_legal, 1)

    over_means_r = (legal_sub.groupby("overid")["tot_runs"].mean() + extra_rate).to_dict()
    over_means_p = legal_sub.groupby("overid")["is_wicket"].mean().to_dict()

    grp = legal_sub.groupby(["overid", "wk_before"])
    counts = grp.size().to_dict()
    raw_r = (grp["tot_runs"].mean() + extra_rate).to_dict()
    raw_p = grp["is_wicket"].mean().to_dict()

    R = np.zeros((21, 11), dtype=float)
    P = np.zeros((21, 11), dtype=float)

    for o in range(1, 21):
        prior_r = over_means_r.get(o, 1.35)
        prior_p = over_means_p.get(o, 0.05)
        for w in range(0, 10):
            n = counts.get((o, w), 0)
            r_obs = raw_r.get((o, w), prior_r)
            p_obs = raw_p.get((o, w), prior_p)

            # Tailender degradation factor
            w_factor = max(0.60, 1.0 - 0.04 * max(0, w - 4))
            p_w_factor = min(2.50, 1.0 + 0.08 * max(0, w - 4))

            cell_prior_r = prior_r * w_factor
            cell_prior_p = min(0.30, prior_p * p_w_factor)

            R[o, w] = (n * r_obs + M * cell_prior_r) / (n + M)
            P[o, w] = (n * p_obs + M * cell_prior_p) / (n + M)

    # Dynamic Programming Table
    V = np.zeros((121, 11), dtype=float)
    for b in range(1, 121):
        o = (120 - b) // 6 + 1
        for w in range(0, 10):
            r = R[o, w]
            p = P[o, w]
            V[b, w] = r + p * V[b - 1, min(10, w + 1)] + (1.0 - p) * V[b - 1, w]

    return V


def compute_run_values(df, config):
    print("--- Computing Run-Expectancy Tables V(b, w) ---")
    V_ipl = build_dp_table(df, comp_group="IPL")
    V_t20i = build_dp_table(df, comp_group="T20I")

    print(f"IPL  -> V(120,0): {V_ipl[120,0]:.1f}, V(60,0): {V_ipl[60,0]:.1f}, V(12,0): {V_ipl[12,0]:.1f}, V(1,0): {V_ipl[1,0]:.2f}")
    print(f"T20I -> V(120,0): {V_t20i[120,0]:.1f}, V(60,0): {V_t20i[60,0]:.1f}, V(12,0): {V_t20i[12,0]:.1f}, V(1,0): {V_t20i[1,0]:.2f}")

    # Vectorized lookups for V(b, w)
    b_curr = np.clip(df["balls_left"].values, 0, 120)
    w_curr = np.clip(df["wk_before"].values, 0, 9)
    leg = df["legal"].values
    b_next = np.clip(b_curr - leg, 0, 120)
    w_next = np.clip(w_curr + 1, 0, 10)

    # Select table based on comp_group (IPL gets V_ipl, others get V_t20i)
    is_ipl_arr = (df["comp_group"] == "IPL").values

    v_curr_ipl = V_ipl[b_curr, w_curr]
    v_next_ipl = V_ipl[b_next, w_curr]
    v_wkt_ipl = V_ipl[b_next, w_next]

    v_curr_t20i = V_t20i[b_curr, w_curr]
    v_next_t20i = V_t20i[b_next, w_curr]
    v_wkt_t20i = V_t20i[b_next, w_next]

    v_curr = np.where(is_ipl_arr, v_curr_ipl, v_curr_t20i)
    v_next = np.where(is_ipl_arr, v_next_ipl, v_next_t20i)
    v_wkt = np.where(is_ipl_arr, v_wkt_ipl, v_wkt_t20i)

    # Time term: expected runs consumed by ball
    time_term = v_next - v_curr
    # If illegal ball, time_term is 0
    time_term = np.where(leg == 1, time_term, 0.0)
    df["time_term"] = time_term

    # Wicket cost (always negative)
    wk_cost = v_wkt - v_next
    df["wk_cost"] = wk_cost

    # 1. Batter RV
    # Striker faces delivery: non-wide
    is_wide = (df["extras_type"] == "wide").values
    striker_faced = ~is_wide
    bat_runs = df["runs"].values

    # Base striker RV
    rv_striker = np.where(striker_faced, bat_runs + time_term, 0.0)

    # Wicket assignment
    is_wkt = (df["is_wicket"] == 1).values
    striker_id = df["striker_id"].values
    non_striker_id = df["non_striker_id"].values
    dismissed_id = df["dismissed_player_id"].values

    striker_dismissed = is_wkt & (dismissed_id == striker_id)
    non_striker_dismissed = is_wkt & (dismissed_id == non_striker_id)

    # Add wk_cost to striker if dismissed
    rv_striker = np.where(striker_dismissed, rv_striker + wk_cost, rv_striker)
    df["rv_bat"] = rv_striker
    df["is_striker_faced"] = striker_faced.astype(int)

    # Non-striker wicket cost (e.g. run-outs where non-striker was out)
    df["non_striker_rv_penalty"] = np.where(non_striker_dismissed, wk_cost, 0.0)

    # 2. Bowler RV
    bowler_runs = df["bowler_runs"].values
    is_bowl_wkt = df["is_bowler_wicket"].values
    rv_bowl = -(bowler_runs + time_term) - np.where(is_bowl_wkt, wk_cost, 0.0)
    df["rv_bowl"] = rv_bowl

    # 3. Team RV
    tot_runs = df["tot_runs"].values
    team_rv = tot_runs + time_term + np.where(is_wkt, wk_cost, 0.0)
    df["team_rv"] = team_rv

    print("--- Verifying Team RV Balances (§7.4) ---")
    valid_mask = (~df["is_dls"]).values
    for cg in ["IPL", "T20I"]:
        cg_mask = valid_mask & (df["comp_group"] == cg).values
        for inn in [1, 2]:
            m = cg_mask & (df["innings"] == inn).values
            mean_team_rv = np.mean(team_rv[m])
            print(f"  {cg} Innings {inn}: Mean Team RV per ball = {mean_team_rv:+.5f}")
            assert abs(mean_team_rv) < 0.05, f"Mean team RV too high: {mean_team_rv}"

    # 4. Pressure Definitions (§4.3)
    p_cfg = config["pressure"]
    li_eff = df["li_eff"].values

    df["is_p_li_15"] = (li_eff >= 1.5).astype(int)
    df["is_p_li_125"] = (li_eff >= 1.25).astype(int)
    df["is_p_li_175"] = (li_eff >= 1.75).astype(int)
    df["is_p_li_20"] = (li_eff >= 2.0).astype(int)

    # P-Chase
    df["is_p_chase"] = (
        (df["innings"] == 2)
        & (df["balls_left"] <= p_cfg["p_chase"]["max_balls_left"])
        & (df["wp_before"] >= p_cfg["p_chase"]["min_wp"])
        & (df["wp_before"] <= p_cfg["p_chase"]["max_wp"])
    ).astype(int)

    # P-Squeeze
    df["is_p_squeeze"] = (
        (df["innings"] == 2)
        & (df["balls_left"] <= p_cfg["p_squeeze"]["max_balls_left"])
        & ((df["rrr_minus_crr"] >= p_cfg["p_squeeze"]["min_rrr_minus_crr"]) | (df["rrr"] >= p_cfg["p_squeeze"]["min_rrr"]))
    ).astype(int)

    # P-Collapse
    df["is_p_collapse"] = (
        ((df["legal_before"] <= p_cfg["p_collapse"]["early_balls"]) & (df["wk_before"] >= p_cfg["p_collapse"]["early_wickets"]))
        | ((df["legal_before"] <= p_cfg["p_collapse"]["mid_balls"]) & (df["wk_before"] >= p_cfg["p_collapse"]["mid_wickets"]))
    ).astype(int)

    # P-Death
    df["is_p_death"] = (
        (df["overid"] >= p_cfg["p_death"]["min_over"])
        & (
            ((df["innings"] == 2) & ((df["wp_before"] - 0.5).abs() <= p_cfg["p_death"]["wp_center_diff"]))
            | (df["innings"] == 1)
        )
    ).astype(int)

    # Primary continuous leverage weight
    gamma = p_cfg.get("gamma", 1.0)
    df["li_weight"] = np.clip(li_eff, p_cfg["min_clip"], p_cfg["max_clip"]) ** gamma

    print("Pressure Definition Frequencies:")
    for col in ["is_p_li_15", "is_p_li_125", "is_p_li_175", "is_p_li_20", "is_p_chase", "is_p_squeeze", "is_p_collapse", "is_p_death"]:
        pct = df[col].mean() * 100
        print(f"  {col}: {pct:.2f}% of balls ({df[col].sum()} deliveries)")

    return df, V_ipl, V_t20i


if __name__ == "__main__":
    cfg = load_config()
    in_path = os.path.join(cfg["paths"]["output_dir"], "ball_state_wp.parquet")
    df = pd.read_parquet(in_path)

    df, V_ipl, V_t20i = compute_run_values(df, cfg)

    out_path = os.path.join(cfg["paths"]["output_dir"], "ball_state_rv.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Step 3 complete! Saved deliveries with Run Values to {out_path}")
