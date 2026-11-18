"""
04_adjust.py - Context Centering, Opposition and Venue Adjustments via Ridge Regression.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §4.5, §4.8, §7.5.
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def perform_adjustments(df, config):
    print("--- 1. Context-Centring within (comp_group x innings x LI-bucket) ---")
    valid_mask = ~df["is_dls"]
    
    # 8-quantiles of effective leverage
    df["li_bin8"] = pd.qcut(df["li_eff"], q=config["adjustments"].get("li_bins", 8), labels=False, duplicates="drop")

    # Batters centring (on striker-faced deliveries)
    striker_mask = valid_mask & (df["is_striker_faced"] == 1)
    bat_grp_mean = df[striker_mask].groupby(["comp_group", "innings", "li_bin8"])["rv_bat"].mean()
    df["rv_bat_c"] = 0.0
    df.loc[striker_mask, "rv_bat_c"] = df.loc[striker_mask, "rv_bat"] - df.loc[striker_mask].set_index(["comp_group", "innings", "li_bin8"]).index.map(bat_grp_mean).values

    # Bowlers centring
    bowl_grp_mean = df[valid_mask].groupby(["comp_group", "innings", "li_bin8"])["rv_bowl"].mean()
    df["rv_bowl_c"] = 0.0
    df.loc[valid_mask, "rv_bowl_c"] = df.loc[valid_mask, "rv_bowl"] - df.loc[valid_mask].set_index(["comp_group", "innings", "li_bin8"]).index.map(bowl_grp_mean).values

    print("Centred RV summary:")
    print(f"  Striker RV centred mean: {df.loc[striker_mask, 'rv_bat_c'].mean():.4f}, std: {df.loc[striker_mask, 'rv_bat_c'].std():.4f}")
    print(f"  Bowler RV centred mean:  {df.loc[valid_mask, 'rv_bowl_c'].mean():.4f}, std: {df.loc[valid_mask, 'rv_bowl_c'].std():.4f}")

    print("--- 2. Fitting Batter Adjustment Ridge Model (Opponent Bowler + Venue) ---")
    bat_df = df[striker_mask].copy()

    enc_s = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_s = enc_s.fit_transform(bat_df[["striker_id"]])

    enc_b = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_b = enc_b.fit_transform(bat_df[["bowler_id"]])

    enc_v = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_v = enc_v.fit_transform(bat_df[["venue_id"]])

    X_bat = sparse.hstack([X_s, X_b, X_v])
    y_bat = bat_df["rv_bat_c"].values

    # Proper regularization for 1600+ features
    alpha = 1000.0
    ridge_bat = Ridge(alpha=alpha, solver="lsqr", random_state=config["seed"])
    ridge_bat.fit(X_bat, y_bat)

    n_s = X_s.shape[1]
    n_b = X_b.shape[1]
    coef_b = ridge_bat.coef_[n_s : n_s + n_b]
    coef_v = ridge_bat.coef_[n_s + n_b :]

    # Map coefficients back to bowlers and venues
    b_ids = [int(c.split("_")[-1]) for c in enc_b.get_feature_names_out(["bowler_id"])]
    bowler_effects = dict(zip(b_ids, coef_b))

    v_ids = [int(c.split("_")[-1]) for c in enc_v.get_feature_names_out(["venue_id"])]
    venue_effects_bat = dict(zip(v_ids, coef_v))

    df["opp_bowler_effect"] = df["bowler_id"].map(bowler_effects).fillna(0.0)
    df["venue_effect_bat"] = df["venue_id"].map(venue_effects_bat).fillna(0.0)
    df["rv_bat_adj"] = np.where(
        striker_mask,
        df["rv_bat_c"] - df["opp_bowler_effect"] - df["venue_effect_bat"],
        0.0
    )

    print("--- 3. Fitting Bowler Adjustment Ridge Model (Opponent Batter + Venue) ---")
    bowl_df = df[valid_mask].copy()

    enc_bowl_b = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_bb = enc_bowl_b.fit_transform(bowl_df[["bowler_id"]])

    enc_bowl_s = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_bs = enc_bowl_s.fit_transform(bowl_df[["striker_id"]])

    enc_bowl_v = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_bv = enc_bowl_v.fit_transform(bowl_df[["venue_id"]])

    X_bowl = sparse.hstack([X_bb, X_bs, X_bv])
    y_bowl = bowl_df["rv_bowl_c"].values

    ridge_bowl = Ridge(alpha=alpha, solver="lsqr", random_state=config["seed"])
    ridge_bowl.fit(X_bowl, y_bowl)

    nb_b = X_bb.shape[1]
    nb_s = X_bs.shape[1]
    coef_bs = ridge_bowl.coef_[nb_b : nb_b + nb_s]
    coef_bv = ridge_bowl.coef_[nb_b + nb_s :]

    s_ids = [int(c.split("_")[-1]) for c in enc_bowl_s.get_feature_names_out(["striker_id"])]
    striker_effects = dict(zip(s_ids, coef_bs))

    bv_ids = [int(c.split("_")[-1]) for c in enc_bowl_v.get_feature_names_out(["venue_id"])]
    venue_effects_bowl = dict(zip(bv_ids, coef_bv))

    df["opp_striker_effect"] = df["striker_id"].map(striker_effects).fillna(0.0)
    df["venue_effect_bowl"] = df["venue_id"].map(venue_effects_bowl).fillna(0.0)
    df["rv_bowl_adj"] = np.where(
        valid_mask,
        df["rv_bowl_c"] - df["opp_striker_effect"] - df["venue_effect_bowl"],
        0.0
    )

    # Sanity audit table
    b_df = pd.DataFrame({"bowler_id": b_ids, "difficulty_coef": coef_b})
    b_map = df[["bowler_id", "bowler_name", "bowling_team"]].drop_duplicates("bowler_id")
    b_df = b_df.merge(b_map, on="bowler_id")
    
    s_df = pd.DataFrame({"striker_id": s_ids, "threat_coef": coef_bs})
    s_map = df[["striker_id", "striker_name", "batting_team"]].drop_duplicates("striker_id")
    s_df = s_df.merge(s_map, on="striker_id")

    audit_path = os.path.join(config["paths"]["output_dir"], "sanity_adjustments.csv")
    sanity = pd.concat([
        b_df.sort_values("difficulty_coef").head(10).assign(type="Toughest Bowlers (Suppress runs)"),
        b_df.sort_values("difficulty_coef", ascending=False).head(10).assign(type="Easiest Bowlers (Leak runs)"),
        s_df.sort_values("threat_coef").head(10).assign(type="Most Dangerous Batters (Hard to bowl to)"),
        s_df.sort_values("threat_coef", ascending=False).head(10).assign(type="Easiest Batters (Free wickets/dots)")
    ], axis=0)
    sanity.to_csv(audit_path, index=False)
    print(f"Saved adjustment sanity audit to {audit_path}")
    print(f"Bowler effects range: [{coef_b.min():.3f}, {coef_b.max():.3f}], std: {coef_b.std():.3f}")
    print(f"Striker effects range: [{coef_bs.min():.3f}, {coef_bs.max():.3f}], std: {coef_bs.std():.3f}")

    return df


if __name__ == "__main__":
    cfg = load_config()
    in_path = os.path.join(cfg["paths"]["output_dir"], "ball_state_rv.parquet")
    df = pd.read_parquet(in_path)

    df = perform_adjustments(df, cfg)

    out_path = os.path.join(cfg["paths"]["output_dir"], "ball_state_adj.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Step 4 complete! Saved adjusted deliveries to {out_path}")
