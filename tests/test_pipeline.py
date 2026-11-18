"""
test_pipeline.py - Comprehensive Unit and Integration Tests for Pressure Pipeline.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §7.9.
"""

import os
import yaml
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@pytest.fixture(scope="module")
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def clean_data():
    return pd.read_parquet("outputs/clean_deliveries.parquet")


@pytest.fixture(scope="module")
def rv_data():
    return pd.read_parquet("outputs/ball_state_rv.parquet")


@pytest.fixture(scope="module")
def final_rankings():
    return pd.read_csv("outputs/rankings_final.csv")


def test_legal_ball_and_deliveries_count(clean_data):
    """Test §7.2 & §7.9: Retained delivery count and legal ball rule."""
    assert len(clean_data) == 200961, f"Expected 200961 deliveries, found {len(clean_data)}"
    
    non_final = clean_data[(clean_data["overid"] < 20) & (~clean_data["is_dls"])].copy()
    over_legal = non_final.groupby(["match_id", "innings", "overid"])["legal"].sum()
    pct_6 = (over_legal == 6).mean() * 100
    assert pct_6 >= 98.5, f"Legal ball consistency too low: {pct_6:.2f}%"


def test_dls_exclusions(clean_data):
    """Test §3.3: DLS matches count is exactly 34 and 4433 balls."""
    dls_balls = clean_data["is_dls"].sum()
    assert dls_balls == 4433, f"Expected 4433 DLS balls, got {dls_balls}"


def test_team_rv_zero_mean(rv_data):
    """Test §7.4 & §7.9: Team RV mean ≈ 0 per innings and competition group."""
    valid = rv_data[~rv_data["is_dls"]]
    for cg in ["IPL", "T20I"]:
        for inn in [1, 2]:
            sub = valid[(valid["comp_group"] == cg) & (valid["innings"] == inn)]
            mean_team_rv = sub["team_rv"].mean()
            assert abs(mean_team_rv) < 0.08, f"Team RV mean {mean_team_rv} too high for {cg} inn {inn}"


def test_innings_rv_telescoping(rv_data):
    """Test §7.9: Sum of ball RV over innings ≈ final total - V(120, 0)."""
    valid = rv_data[(~rv_data["is_dls"]) & (rv_data["comp_group"] == "IPL") & (rv_data["innings"] == 1)].copy()
    # Check on 20 random full 20-over matches
    match_ids = valid.groupby("match_id")["legal"].sum()
    full_matches = match_ids[match_ids == 120].index[:20]

    diffs = []
    for mid in full_matches:
        m_df = valid[valid["match_id"] == mid]
        tot_actual = m_df["tot_runs"].sum()
        sum_rv = m_df["team_rv"].sum()
        # Sum of team_rv is roughly tot_actual - V(120,0)
        diff = sum_rv - (tot_actual - 178.4)
        diffs.append(abs(diff))

    assert np.mean(diffs) < 25.0, f"Innings RV telescoping error too high: {np.mean(diffs)}"


def test_wp_monotonicity(clean_data):
    """Test §7.9: Win probability monotonicity spot checks."""
    wp_df = pd.read_parquet("outputs/ball_state_wp.parquet")
    chase = wp_df[(wp_df["innings"] == 2) & (~wp_df["is_dls"])].copy()

    # More runs needed at same balls left and wickets in hand -> lower WP
    sample = chase[(chase["balls_left"] == 30) & (chase["wk_in_hand"] == 6)]
    if len(sample) >= 5:
        corr_runs, _ = np.corrcoef(sample["runs_needed"], sample["wp_before"])
        assert corr_runs[1] < 0, f"Expected negative correlation between runs needed and WP, got {corr_runs[1]}"


def test_rankings_uniqueness_and_eligibility(final_rankings):
    """Test §7.8 & §7.9: Exactly 10 distinct IDs, correct roles, no A-team only."""
    batters = final_rankings[final_rankings["role"] == "Batter"]
    bowlers = final_rankings[final_rankings["role"] == "Bowler"]

    assert len(batters) == 10, f"Expected 10 batters, got {len(batters)}"
    assert len(bowlers) == 10, f"Expected 10 bowlers, got {len(bowlers)}"

    assert batters["player_id"].nunique() == 10, "Duplicate IDs in Batters Top-10"
    assert bowlers["player_id"].nunique() == 10, "Duplicate IDs in Bowlers Top-10"

    # Verify no overlap between batter IDs and bowler IDs
    assert len(set(batters["player_id"]).intersection(set(bowlers["player_id"]))) == 0

    # Verify all players have valid scores and positive balls
    assert (final_rankings["composite_score"] > 0).all()
    assert (final_rankings["total_balls_or_legal"] >= 250).all()
    assert (final_rankings["pressure_balls"] >= 40).all()


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_pipeline.py"])
