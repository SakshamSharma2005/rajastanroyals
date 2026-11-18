"""
01_load_clean.py - Data loading, cleaning, state reconstruction, and tier assignment.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §7.2.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_clean_data(config=None):
    if config is None:
        config = load_config()

    # 1. Read files
    bbb = pd.read_csv(config["paths"]["ball_by_ball"])
    matches = pd.read_csv(config["paths"]["matches"])
    comps = pd.read_csv(config["paths"]["competitions"])
    teams = pd.read_csv(config["paths"]["teams"])
    mapping = pd.read_csv(config["paths"]["mapping"])
    p11 = pd.read_csv(config["paths"]["playing_eleven"])

    # 2. Team lookup
    id_to_team = teams.drop_duplicates("team_id").set_index("team_id")["team_name"].to_dict()

    # 3. Matches and competitions join
    matches["match_date_dt"] = pd.to_datetime(matches["match_date"])
    matches["year"] = matches["match_date_dt"].dt.year

    # DLS flag
    dls_mask = (
        matches["first_inn_revised_overs"].notna()
        | matches["second_inn_revised_overs"].notna()
        | matches["revised_target"].notna()
    )
    matches["is_dls"] = dls_mask

    # Competitions merge
    matches = matches.merge(
        comps[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left"
    )

    # Derive is_ipl and comp_group
    is_ipl = matches["comp_name"].str.contains("IPL", case=False, na=False) | matches["comp_id"].isin([1358, 1715, 2193, 2725, 3298])
    matches["is_ipl"] = is_ipl

    a_team_comps = config.get("a_team_comps", [2670, 3289])
    is_a_team = matches["comp_id"].isin(a_team_comps) | matches["comp_name"].str.contains("Emerging|Rising Stars", case=False, na=False)

    comp_group = np.where(is_ipl, "IPL", np.where(is_a_team, "A_TEAM", "T20I"))
    matches["comp_group"] = comp_group

    # Recency weight (half-life = 2 years)
    max_date = matches["match_date_dt"].max()
    matches["age_years"] = (max_date - matches["match_date_dt"]).dt.total_seconds() / (365.25 * 86400.0)
    half_life = config["ratings"]["recency_half_life_years"]
    matches["recency_weight"] = 0.5 ** (matches["age_years"] / half_life)

    # Match tier and tier_weight
    full_members = set(config["full_members"])
    associates = set(config["associates"])

    def get_match_tier(row):
        if row["comp_group"] == "IPL":
            return "ipl", config["team_tiers"]["ipl"]
        elif row["comp_group"] == "A_TEAM":
            return "a_team", config["team_tiers"]["a_team"]
        else:
            # T20I
            home = str(row["home_team"])
            away = str(row["away_team"])
            if home in associates or away in associates:
                return "associate", config["team_tiers"]["associate"]
            elif home in full_members and away in full_members:
                return "full_member", config["team_tiers"]["full_member"]
            else:
                return "associate", config["team_tiers"]["associate"]

    tier_info = matches.apply(get_match_tier, axis=1)
    matches["tier"] = [t[0] for t in tier_info]
    matches["tier_weight"] = [t[1] for t in tier_info]

    # 4. Clean ball_by_ball: drop super overs (innings >= 3)
    initial_len = len(bbb)
    bbb = bbb[bbb["innings"].isin([1, 2])].copy()
    dropped_so = initial_len - len(bbb)

    # 5. Extras mapping
    ex_map = config["extras_run_values"]
    bbb["ex_runs"] = bbb["extras_type"].map(ex_map).fillna(0.0)
    bbb["tot_runs"] = bbb["runs"] + bbb["ex_runs"]

    # Bowler charged runs: bat_runs + 1.0 for wide or Overstep
    bowler_charged_extras = config.get("bowler_charged_extras", ["wide", "Overstep"])
    bbb["bowler_runs"] = bbb["runs"] + np.where(bbb["extras_type"].isin(bowler_charged_extras), 1.0, 0.0)

    # Legal ball definition: not wide, Overstep, or other
    illegal_extras = config.get("illegal_extras", ["wide", "Overstep", "other"])
    bbb["legal"] = (~bbb["extras_type"].isin(illegal_extras)).astype(int)

    # Sort strictly by match_id, innings, innings_ball_num
    bbb.sort_values(by=["match_id", "innings", "innings_ball_num"], inplace=True)
    bbb.reset_index(drop=True, inplace=True)

    # 6. Reconstruct state before each ball
    grp = bbb.groupby(["match_id", "innings"])
    bbb["legal_before"] = grp["legal"].cumsum() - bbb["legal"]
    bbb["balls_left"] = np.clip(120 - bbb["legal_before"], 0, 120)
    bbb["runs_before"] = grp["tot_runs"].cumsum() - bbb["tot_runs"]
    bbb["wk_before"] = grp["is_wicket"].cumsum() - bbb["is_wicket"]
    bbb["wk_in_hand"] = np.clip(10 - bbb["wk_before"], 0, 10)

    # Over phases
    # PP: 1-6, M1: 7-10, M2: 11-15, D: 16-20
    over = bbb["overid"]
    bbb["phase"] = np.where(
        over <= 6, "PP",
        np.where(over <= 10, "M1",
                 np.where(over <= 15, "M2", "D"))
    )

    # Total innings 1 score for target calculation
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["tot_runs"].sum().rename("inn1_total")
    bbb = bbb.merge(inn1_totals, on="match_id", how="left")
    bbb["inn1_total"] = bbb["inn1_total"].fillna(0.0)

    # Innings 2 chase variables
    bbb["target"] = np.where(bbb["innings"] == 2, bbb["inn1_total"] + 1.0, 0.0)
    bbb["runs_needed"] = np.where(bbb["innings"] == 2, np.maximum(0.0, bbb["target"] - bbb["runs_before"]), 0.0)

    # RRR and CRR
    overs_left = np.maximum(bbb["balls_left"] / 6.0, 1.0 / 6.0)
    bbb["rrr"] = np.where(bbb["innings"] == 2, bbb["runs_needed"] / overs_left, 0.0)
    overs_bowled = np.maximum(bbb["legal_before"] / 6.0, 1.0 / 6.0)
    bbb["crr"] = bbb["runs_before"] / overs_bowled
    bbb["rrr_minus_crr"] = np.where(bbb["innings"] == 2, bbb["rrr"] - bbb["crr"], 0.0)

    # 7. Merge match attributes
    match_cols = [
        "match_id", "comp_id", "comp_name", "comp_group", "is_ipl", "year",
        "match_date", "match_date_dt", "stage", "venue_id", "venue_country",
        "knock_out", "win_team_id", "result", "is_dls", "tier", "tier_weight",
        "recency_weight"
    ]
    bbb = bbb.merge(matches[match_cols], on="match_id", how="left")

    # Team names
    bbb["batting_team"] = bbb["batting_team_id"].map(id_to_team)
    bbb["bowling_team"] = bbb["bowling_team_id"].map(id_to_team)

    # Outcome label: bat_won
    # If win_team_id is NaN (abandoned, no result, tie without super over winner), bat_won is NaN
    bbb["bat_won"] = np.where(
        bbb["win_team_id"].isna(),
        np.nan,
        (bbb["batting_team_id"] == bbb["win_team_id"]).astype(float)
    )

    # Bowler credited wicket flag
    bowler_wkts = config.get("bowler_dismissal_types", ["Bowled", "Caught", "LBW", "Stump", "Hit Wicket"])
    bbb["is_bowler_wicket"] = (bbb["is_wicket"] == 1) & (bbb["dismissal_type"].isin(bowler_wkts))

    return bbb, matches, mapping, p11


def run_audits(df, matches):
    print("=== DATA AUDIT RESULTS (§7.2) ===")
    print(f"Total deliveries retained: {len(df)}")
    assert len(df) == 200961, f"Expected 200961 deliveries, got {len(df)}"

    dls_count = df["is_dls"].sum()
    print(f"DLS deliveries: {dls_count} across {matches['is_dls'].sum()} matches")
    assert dls_count == 4433, f"Expected 4433 DLS deliveries, got {dls_count}"

    # Over legal ball check
    non_final = df[(df["overid"] < 20) & (~df["is_dls"])].copy()
    over_legal = non_final.groupby(["match_id", "innings", "overid"])["legal"].sum()
    pct_6 = (over_legal == 6).mean() * 100
    print(f"Percentage of completed non-final overs with exactly 6 legal balls: {pct_6:.2f}%")

    print(f"Competitions breakdown: {df['comp_group'].value_counts().to_dict()}")
    print(f"Tier breakdown: {df['tier'].value_counts().to_dict()}")
    print("=== AUDIT PASSED ===")


if __name__ == "__main__":
    cfg = load_config()
    df, matches, mapping, p11 = load_and_clean_data(cfg)
    run_audits(df, matches)
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "clean_deliveries.parquet"), index=False)
    print(f"Saved clean deliveries to {os.path.join(out_dir, 'clean_deliveries.parquet')}")
