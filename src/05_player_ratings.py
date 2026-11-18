"""
05_player_ratings.py - Leverage-Weighted Means, Empirical Bayes Shrinkage,
Baseline Resilience, Match Consistency, and Composite Ranking.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §4.6, §4.7, §7.6.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_player_metadata(df, mapping_path="competition_team_player_mapping.csv"):
    mapping = pd.read_csv(mapping_path)
    intl_teams = {
        "England", "West Indies", "Australia", "Sri Lanka", "South Africa",
        "Bangladesh", "Zimbabwe", "Afghanistan", "Ireland", "New Zealand",
        "Pakistan", "Hong Kong", "Netherlands", "United Arab Emirates",
        "Singapore", "Kuwait", "Scotland", "Canada", "United States Of America",
        "Oman", "Nepal", "Japan", "Pakistan A", "Bangladesh A", "Sri Lanka A",
        "Afghanistan A"
    }

    p_teams = mapping.groupby("player_id")["team_name"].apply(lambda s: sorted(list(set(s)))).to_dict()
    p_names = mapping.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()

    # Also check names in ball_by_ball
    for pid, name in zip(df["striker_id"], df["striker_name"]):
        if pid not in p_names:
            p_names[pid] = name
    for pid, name in zip(df["bowler_id"], df["bowler_name"]):
        if pid not in p_names:
            p_names[pid] = name

    def get_nationality(pid):
        teams = set(p_teams.get(pid, []))
        if any(t in teams for t in ["India", "India A"]):
            return "Indian"
        if any(t in intl_teams for t in teams):
            return "Overseas"
        return "Indian"  # Domestic Indian players in IPL

    def is_rr_player(pid):
        teams = set(p_teams.get(pid, []))
        return "Rajasthan Royals" in teams

    # Check for duplicate names
    name_counts = pd.Series(p_names).value_counts()
    dup_names = set(name_counts[name_counts > 1].index)

    meta = {}
    for pid, name in p_names.items():
        teams = p_teams.get(pid, [])
        nat = get_nationality(pid)
        is_rr = is_rr_player(pid)
        disp_name = name
        if name in dup_names:
            team_str = teams[0] if len(teams) > 0 else nat
            disp_name = f"{name} ({team_str})"

        meta[pid] = {
            "player_id": pid,
            "player_name": name,
            "display_name": disp_name,
            "nationality": nat,
            "teams": ", ".join(teams) if len(teams) > 0 else "N/A",
            "is_rr": is_rr
        }

    return meta


def compute_batter_ratings(df, meta, config):
    print("--- Computing Batter Ratings ---")
    valid = df[(~df["is_dls"]) & (df["is_striker_faced"] == 1)].copy()

    # Base weights
    gamma = config["pressure"].get("gamma", 1.0)
    valid["w_pressure"] = valid["tier_weight"] * valid["recency_weight"] * (np.clip(valid["li_eff"], 0.25, 4.0) ** gamma)
    valid["w_base"] = valid["tier_weight"] * valid["recency_weight"] * (valid["li_eff"] < config["ratings"]["baseline_li_cutoff"]).astype(float)

    # 1. Player Level Aggregation
    grp = valid.groupby("striker_id")
    stats = grp.agg(
        n_balls=("runs", "count"),
        n_matches=("match_id", "nunique"),
        n_pressure_125=("is_p_li_125", "sum"),
        n_pressure_15=("is_p_li_15", "sum"),
        max_year=("year", "max"),
        total_runs=("runs", "sum"),
        total_legal=("legal", "sum"),
        w_press_sum=("w_pressure", "sum"),
        w_press_sq_sum=("w_pressure", lambda x: np.sum(x**2)),
        w_press_val=("w_pressure", lambda x: np.sum(x * valid.loc[x.index, "rv_bat_adj"])),
        w_base_sum=("w_base", "sum"),
        w_base_sq_sum=("w_base", lambda x: np.sum(x**2)),
        w_base_val=("w_base", lambda x: np.sum(x * valid.loc[x.index, "rv_bat_adj"])),
        # Splits
        ipl_balls=("is_ipl", "sum"),
        pp_balls=("phase", lambda x: np.sum(x == "PP")),
        death_balls=("phase", lambda x: np.sum(x == "D")),
        pressure_runs=("runs", lambda x: np.sum(x[valid.loc[x.index, "is_p_li_125"] == 1])),
        pressure_legal=("legal", lambda x: np.sum(x[valid.loc[x.index, "is_p_li_125"] == 1]))
    ).reset_index()

    stats.rename(columns={"striker_id": "player_id"}, inplace=True)

    # Effective sample sizes and weighted means
    stats["n_eff_press"] = (stats["w_press_sum"]**2) / np.maximum(stats["w_press_sq_sum"], 1e-6)
    stats["m_press"] = stats["w_press_val"] / np.maximum(stats["w_press_sum"], 1e-6)

    stats["n_eff_base"] = (stats["w_base_sum"]**2) / np.maximum(stats["w_base_sq_sum"], 1e-6)
    stats["m_base"] = stats["w_base_val"] / np.maximum(stats["w_base_sum"], 1e-6)

    # Contextual stats
    stats["raw_sr"] = (stats["total_runs"] / np.maximum(stats["total_legal"], 1)) * 100.0
    stats["pressure_sr"] = (stats["pressure_runs"] / np.maximum(stats["pressure_legal"], 1)) * 100.0
    stats["pct_death"] = (stats["death_balls"] / np.maximum(stats["n_balls"], 1)) * 100.0
    stats["pct_pp"] = (stats["pp_balls"] / np.maximum(stats["n_balls"], 1)) * 100.0
    stats["pct_ipl"] = (stats["ipl_balls"] / np.maximum(stats["n_balls"], 1)) * 100.0

    # Eligibility filter
    b_cfg = config["eligibility"]["batters"]
    stats["is_eligible"] = (
        (stats["n_balls"] >= b_cfg["min_balls_faced"])
        & (stats["n_pressure_125"] >= b_cfg["min_high_pressure_balls"])
        & (stats["n_matches"] >= b_cfg["min_matches"])
        & (stats["max_year"] >= b_cfg["active_year_min"])
    )

    el = stats[stats["is_eligible"]].copy()
    print(f"Eligible Batters count: {len(el)} (out of {len(stats)} total)")

    # 2. Empirical Bayes Shrinkage for Pressure Level
    sigma2 = np.var(valid["rv_bat_adj"])
    mean_samp_var = np.mean(sigma2 / np.maximum(el["n_eff_press"], 1.0))
    var_m = np.var(el["m_press"], ddof=1)
    tau2 = max(0.005, var_m - mean_samp_var)

    shrink_press = tau2 / (tau2 + sigma2 / np.maximum(stats["n_eff_press"], 1e-6))
    stats["post_press"] = stats["m_press"] * shrink_press
    stats["post_press_sd"] = np.sqrt((tau2 * (sigma2 / np.maximum(stats["n_eff_press"], 1e-6))) / (tau2 + sigma2 / np.maximum(stats["n_eff_press"], 1e-6)))
    stats["lcb_press"] = stats["post_press"] - config["ratings"].get("conservative_k", 1.0) * stats["post_press_sd"]

    # Empirical Bayes Shrinkage for Baseline
    mean_samp_var_b = np.mean(sigma2 / np.maximum(el["n_eff_base"], 1.0))
    var_m_b = np.var(el["m_base"], ddof=1)
    tau2_b = max(0.005, var_m_b - mean_samp_var_b)

    shrink_base = tau2_b / (tau2_b + sigma2 / np.maximum(stats["n_eff_base"], 1e-6))
    stats["post_base"] = stats["m_base"] * shrink_base
    stats["post_base_sd"] = np.sqrt((tau2_b * (sigma2 / np.maximum(stats["n_eff_base"], 1e-6))) / (tau2_b + sigma2 / np.maximum(stats["n_eff_base"], 1e-6)))

    # Resilience: Delta vs own baseline
    stats["resilience"] = stats["post_press"] - stats["post_base"]
    stats["resilience_sd"] = np.sqrt(stats["post_press_sd"]**2 + stats["post_base_sd"]**2)

    # 3. Match-Level Consistency
    # Matches with >= 6 pressure balls
    min_mpb = config["ratings"].get("min_match_pressure_balls_batter", 6)
    press_balls_df = valid[valid["li_eff"] >= 1.0].copy()
    m_grp = press_balls_df.groupby(["striker_id", "match_id"]).agg(
        n_mpb=("runs", "count"),
        match_rv=("rv_bat_adj", "sum")
    ).reset_index()
    m_grp = m_grp[m_grp["n_mpb"] >= min_mpb].copy()
    m_grp["hit"] = (m_grp["match_rv"] >= 0.0).astype(float)

    match_stats = m_grp.groupby("striker_id").agg(
        n_press_matches=("match_id", "count"),
        hit_rate=("hit", "mean"),
        p20_rv=("match_rv", lambda x: np.percentile(x, 20) if len(x) >= 3 else np.min(x))
    ).reset_index()
    match_stats.rename(columns={"striker_id": "player_id"}, inplace=True)

    stats = stats.merge(match_stats, on="player_id", how="left")
    stats["n_press_matches"] = stats["n_press_matches"].fillna(0)
    # Shrunk hit rate (prior = 0.50, weight = 5 matches)
    stats["hit_rate_shrunk"] = (stats["n_press_matches"] * stats["hit_rate"].fillna(0.5) + 5.0 * 0.5) / (stats["n_press_matches"] + 5.0)
    stats["p20_rv"] = stats["p20_rv"].fillna(0.0)

    # 4. Composite Score Standardisation (on eligible pool)
    el_mask = stats["is_eligible"]
    mean_post = stats.loc[el_mask, "post_press"].mean()
    std_post = stats.loc[el_mask, "post_press"].std()
    stats["z_level"] = (stats["post_press"] - mean_post) / std_post

    mean_res = stats.loc[el_mask, "resilience"].mean()
    std_res = stats.loc[el_mask, "resilience"].std()
    stats["z_resilience"] = (stats["resilience"] - mean_res) / std_res

    # Consistency composite: blend shrunk hit rate and p20
    z_hr = (stats["hit_rate_shrunk"] - stats.loc[el_mask, "hit_rate_shrunk"].mean()) / stats.loc[el_mask, "hit_rate_shrunk"].std()
    z_p20 = (stats["p20_rv"] - stats.loc[el_mask, "p20_rv"].mean()) / stats.loc[el_mask, "p20_rv"].std()
    stats["z_consistency"] = 0.50 * z_hr + 0.50 * z_p20

    w_comp = config["ratings"]["composite_weights"]
    stats["composite_score"] = (
        w_comp["level"] * stats["z_level"]
        + w_comp["resilience"] * stats["z_resilience"]
        + w_comp["consistency"] * stats["z_consistency"]
    )
    
    # Conservative composite
    stats["conservative_score"] = stats["composite_score"] - config["ratings"].get("conservative_k", 1.0) * (stats["post_press_sd"] / std_post)

    # Merge metadata
    meta_df = pd.DataFrame.from_dict(meta, orient="index")
    stats = stats.merge(meta_df, on="player_id", how="left")

    # 5 Alternate definitions for robustness (§4.3)
    alt_defs = [
        ("rank_p_li_175", "is_p_li_175"),
        ("rank_p_chase", "is_p_chase"),
        ("rank_p_squeeze", "is_p_squeeze"),
        ("rank_p_collapse", "is_p_collapse"),
        ("rank_p_death", "is_p_death"),
    ]
    for r_col, mask_col in alt_defs:
        sub = valid[valid[mask_col] == 1].copy()
        w = sub["tier_weight"] * sub["recency_weight"] * np.clip(sub["li_eff"], 0.25, 4.0)
        sc = (sub["rv_bat_adj"] * w).groupby(sub["striker_id"]).sum() / np.maximum(w.groupby(sub["striker_id"]).sum(), 1e-6)
        sc_el = sc.loc[sc.index.isin(stats.loc[el_mask, "player_id"])]
        stats[r_col] = stats["player_id"].map(sc_el.rank(ascending=False, method="min"))

    r_cols = [r_col for r_col, _ in alt_defs]
    stats["n_alt_top15"] = (stats[r_cols] <= 15).sum(axis=1)

    # Rank eligible players
    stats["rank_composite"] = np.nan
    stats.loc[el_mask, "rank_composite"] = stats.loc[el_mask, "composite_score"].rank(ascending=False, method="min").astype(int)

    stats["rank_conservative"] = np.nan
    stats.loc[el_mask, "rank_conservative"] = stats.loc[el_mask, "conservative_score"].rank(ascending=False, method="min").astype(int)

    stats["role"] = "Batter"
    stats.sort_values(by="composite_score", ascending=False, inplace=True)
    return stats


def compute_bowler_ratings(df, meta, config):
    print("--- Computing Bowler Ratings ---")
    valid = df[~df["is_dls"]].copy()

    gamma = config["pressure"].get("gamma", 1.0)
    valid["w_pressure"] = valid["tier_weight"] * valid["recency_weight"] * (np.clip(valid["li_eff"], 0.25, 4.0) ** gamma)
    valid["w_base"] = valid["tier_weight"] * valid["recency_weight"] * (valid["li_eff"] < config["ratings"]["baseline_li_cutoff"]).astype(float)

    # 1. Player Level Aggregation
    grp = valid.groupby("bowler_id")
    stats = grp.agg(
        n_legal=("legal", "sum"),
        n_matches=("match_id", "nunique"),
        n_pressure_125=("is_p_li_125", "sum"),
        n_pressure_15=("is_p_li_15", "sum"),
        max_year=("year", "max"),
        total_runs_conceded=("bowler_runs", "sum"),
        total_wkts=("is_bowler_wicket", "sum"),
        w_press_sum=("w_pressure", "sum"),
        w_press_sq_sum=("w_pressure", lambda x: np.sum(x**2)),
        w_press_val=("w_pressure", lambda x: np.sum(x * valid.loc[x.index, "rv_bowl_adj"])),
        w_base_sum=("w_base", "sum"),
        w_base_sq_sum=("w_base", lambda x: np.sum(x**2)),
        w_base_val=("w_base", lambda x: np.sum(x * valid.loc[x.index, "rv_bowl_adj"])),
        ipl_balls=("is_ipl", "sum"),
        pp_balls=("phase", lambda x: np.sum(x == "PP")),
        death_balls=("phase", lambda x: np.sum(x == "D")),
        pressure_runs=("bowler_runs", lambda x: np.sum(x[valid.loc[x.index, "is_p_li_125"] == 1])),
        pressure_legal=("legal", lambda x: np.sum(x[valid.loc[x.index, "is_p_li_125"] == 1])),
        pressure_wkts=("is_bowler_wicket", lambda x: np.sum(x[valid.loc[x.index, "is_p_li_125"] == 1]))
    ).reset_index()

    stats.rename(columns={"bowler_id": "player_id"}, inplace=True)

    # Effective sample sizes and weighted means
    stats["n_eff_press"] = (stats["w_press_sum"]**2) / np.maximum(stats["w_press_sq_sum"], 1e-6)
    stats["m_press"] = stats["w_press_val"] / np.maximum(stats["w_press_sum"], 1e-6)

    stats["n_eff_base"] = (stats["w_base_sum"]**2) / np.maximum(stats["w_base_sq_sum"], 1e-6)
    stats["m_base"] = stats["w_base_val"] / np.maximum(stats["w_base_sum"], 1e-6)

    # Contextual stats
    stats["raw_econ"] = (stats["total_runs_conceded"] / np.maximum(stats["n_legal"], 1)) * 6.0
    stats["pressure_econ"] = (stats["pressure_runs"] / np.maximum(stats["pressure_legal"], 1)) * 6.0
    stats["raw_sr"] = stats["n_legal"] / np.maximum(stats["total_wkts"], 1)
    stats["pct_death"] = (stats["death_balls"] / np.maximum(stats["n_legal"], 1)) * 100.0
    stats["pct_pp"] = (stats["pp_balls"] / np.maximum(stats["n_legal"], 1)) * 100.0
    stats["pct_ipl"] = (stats["ipl_balls"] / np.maximum(stats["n_legal"], 1)) * 100.0

    # Eligibility filter (measured uniformly in legal balls)
    b_cfg = config["eligibility"]["bowlers"]
    stats["is_eligible"] = (
        (stats["n_legal"] >= b_cfg["min_legal_balls"])
        & (stats["pressure_legal"] >= b_cfg["min_high_pressure_balls"])
        & (stats["n_matches"] >= b_cfg["min_matches"])
        & (stats["max_year"] >= b_cfg["active_year_min"])
    )

    el = stats[stats["is_eligible"]].copy()
    print(f"Eligible Bowlers count: {len(el)} (out of {len(stats)} total)")

    # 2. Empirical Bayes Shrinkage for Bowler Level
    sigma2 = np.var(valid["rv_bowl_adj"])
    mean_samp_var = np.mean(sigma2 / np.maximum(el["n_eff_press"], 1.0))
    var_m = np.var(el["m_press"], ddof=1)
    tau2 = max(0.005, var_m - mean_samp_var)

    shrink_press = tau2 / (tau2 + sigma2 / np.maximum(stats["n_eff_press"], 1e-6))
    stats["post_press"] = stats["m_press"] * shrink_press
    stats["post_press_sd"] = np.sqrt((tau2 * (sigma2 / np.maximum(stats["n_eff_press"], 1e-6))) / (tau2 + sigma2 / np.maximum(stats["n_eff_press"], 1e-6)))
    stats["lcb_press"] = stats["post_press"] - config["ratings"].get("conservative_k", 1.0) * stats["post_press_sd"]

    # Baseline EB
    mean_samp_var_b = np.mean(sigma2 / np.maximum(el["n_eff_base"], 1.0))
    var_m_b = np.var(el["m_base"], ddof=1)
    tau2_b = max(0.005, var_m_b - mean_samp_var_b)

    shrink_base = tau2_b / (tau2_b + sigma2 / np.maximum(stats["n_eff_base"], 1e-6))
    stats["post_base"] = stats["m_base"] * shrink_base
    stats["post_base_sd"] = np.sqrt((tau2_b * (sigma2 / np.maximum(stats["n_eff_base"], 1e-6))) / (tau2_b + sigma2 / np.maximum(stats["n_eff_base"], 1e-6)))

    # Resilience
    stats["resilience"] = stats["post_press"] - stats["post_base"]
    stats["resilience_sd"] = np.sqrt(stats["post_press_sd"]**2 + stats["post_base_sd"]**2)

    # 3. Match Consistency
    min_mpb = config["ratings"].get("min_match_pressure_balls_bowler", 6)
    press_balls_df = valid[valid["li_eff"] >= 1.0].copy()
    m_grp = press_balls_df.groupby(["bowler_id", "match_id"]).agg(
        n_mpb=("legal", "sum"),
        match_rv=("rv_bowl_adj", "sum")
    ).reset_index()
    m_grp = m_grp[m_grp["n_mpb"] >= min_mpb].copy()
    m_grp["hit"] = (m_grp["match_rv"] >= 0.0).astype(float)

    match_stats = m_grp.groupby("bowler_id").agg(
        n_press_matches=("match_id", "count"),
        hit_rate=("hit", "mean"),
        p20_rv=("match_rv", lambda x: np.percentile(x, 20) if len(x) >= 3 else np.min(x))
    ).reset_index()
    match_stats.rename(columns={"bowler_id": "player_id"}, inplace=True)

    stats = stats.merge(match_stats, on="player_id", how="left")
    stats["n_press_matches"] = stats["n_press_matches"].fillna(0)
    stats["hit_rate_shrunk"] = (stats["n_press_matches"] * stats["hit_rate"].fillna(0.5) + 5.0 * 0.5) / (stats["n_press_matches"] + 5.0)
    stats["p20_rv"] = stats["p20_rv"].fillna(0.0)

    # 4. Composite Score Standardisation
    el_mask = stats["is_eligible"]
    mean_post = stats.loc[el_mask, "post_press"].mean()
    std_post = stats.loc[el_mask, "post_press"].std()
    stats["z_level"] = (stats["post_press"] - mean_post) / std_post

    mean_res = stats.loc[el_mask, "resilience"].mean()
    std_res = stats.loc[el_mask, "resilience"].std()
    stats["z_resilience"] = (stats["resilience"] - mean_res) / std_res

    z_hr = (stats["hit_rate_shrunk"] - stats.loc[el_mask, "hit_rate_shrunk"].mean()) / stats.loc[el_mask, "hit_rate_shrunk"].std()
    z_p20 = (stats["p20_rv"] - stats.loc[el_mask, "p20_rv"].mean()) / stats.loc[el_mask, "p20_rv"].std()
    stats["z_consistency"] = 0.50 * z_hr + 0.50 * z_p20

    w_comp = config["ratings"]["composite_weights"]
    stats["composite_score"] = (
        w_comp["level"] * stats["z_level"]
        + w_comp["resilience"] * stats["z_resilience"]
        + w_comp["consistency"] * stats["z_consistency"]
    )
    stats["conservative_score"] = stats["composite_score"] - config["ratings"].get("conservative_k", 1.0) * (stats["post_press_sd"] / std_post)

    meta_df = pd.DataFrame.from_dict(meta, orient="index")
    stats = stats.merge(meta_df, on="player_id", how="left")

    # 5 Alternate definitions for robustness (§4.3)
    alt_defs = [
        ("rank_p_li_175", "is_p_li_175"),
        ("rank_p_chase", "is_p_chase"),
        ("rank_p_squeeze", "is_p_squeeze"),
        ("rank_p_collapse", "is_p_collapse"),
        ("rank_p_death", "is_p_death"),
    ]
    for r_col, mask_col in alt_defs:
        sub = valid[valid[mask_col] == 1].copy()
        w = sub["tier_weight"] * sub["recency_weight"] * np.clip(sub["li_eff"], 0.25, 4.0)
        sc = (sub["rv_bowl_adj"] * w).groupby(sub["bowler_id"]).sum() / np.maximum(w.groupby(sub["bowler_id"]).sum(), 1e-6)
        sc_el = sc.loc[sc.index.isin(stats.loc[el_mask, "player_id"])]
        stats[r_col] = stats["player_id"].map(sc_el.rank(ascending=False, method="min"))

    r_cols = [r_col for r_col, _ in alt_defs]
    stats["n_alt_top15"] = (stats[r_cols] <= 15).sum(axis=1)

    stats["rank_composite"] = np.nan
    stats.loc[el_mask, "rank_composite"] = stats.loc[el_mask, "composite_score"].rank(ascending=False, method="min").astype(int)

    stats["rank_conservative"] = np.nan
    stats.loc[el_mask, "rank_conservative"] = stats.loc[el_mask, "conservative_score"].rank(ascending=False, method="min").astype(int)

    stats["role"] = "Bowler"
    stats.sort_values(by="composite_score", ascending=False, inplace=True)
    return stats


if __name__ == "__main__":
    cfg = load_config()
    in_path = os.path.join(cfg["paths"]["output_dir"], "ball_state_adj.parquet")
    df = pd.read_parquet(in_path)

    meta = build_player_metadata(df, cfg["paths"]["mapping"])

    bat_ratings = compute_batter_ratings(df, meta, cfg)
    bowl_ratings = compute_bowler_ratings(df, meta, cfg)

    out_dir = cfg["paths"]["output_dir"]
    bat_ratings.to_csv(os.path.join(out_dir, "rankings_batters_all.csv"), index=False)
    bowl_ratings.to_csv(os.path.join(out_dir, "rankings_bowlers_all.csv"), index=False)

    print("\n=== TOP 10 BATTERS UNDER PRESSURE (Primary Ranking) ===")
    top10_bat = bat_ratings[bat_ratings["is_eligible"]].head(10)
    print(top10_bat[["rank_composite", "player_id", "display_name", "nationality", "n_balls", "n_pressure_125", "composite_score", "post_press", "resilience", "hit_rate_shrunk", "raw_sr", "pressure_sr"]])

    print("\n=== TOP 10 BOWLERS UNDER PRESSURE (Primary Ranking) ===")
    top10_bowl = bowl_ratings[bowl_ratings["is_eligible"]].head(10)
    print(top10_bowl[["rank_composite", "player_id", "display_name", "nationality", "n_legal", "n_pressure_125", "composite_score", "post_press", "resilience", "hit_rate_shrunk", "raw_econ", "pressure_econ"]])

    print("\nStep 5 complete! Saved full ratings to outputs/rankings_*.csv")
