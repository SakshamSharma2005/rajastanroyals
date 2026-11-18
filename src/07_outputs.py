"""
07_outputs.py - Final Selection, RR Fit Notes, Formatted CSV Exports,
Dropdown Pick File Generation, and Summary Figures.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §7.8, §7.10, §1.5.
"""

import os
import shutil
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_final_outputs(config):
    print("--- Generating Final Rankings and Selection Outputs ---")
    out_dir = config["paths"]["output_dir"]
    stab = pd.read_csv(os.path.join(out_dir, "stability.csv"))

    # Separate eligible batters and bowlers
    bat_df = stab[stab["role"] == "Batter"].sort_values("rank_composite").head(10).copy()
    bowl_df = stab[stab["role"] == "Bowler"].sort_values("rank_composite").head(10).copy()

    # Reset ranks 1 to 10
    bat_df["rank"] = range(1, 11)
    bowl_df["rank"] = range(1, 11)

    # Clean nationalities
    overseas_override = {
        781: "Overseas",    # Sunil Narine (West Indies)
        6396: "Overseas",   # Tim David (Australia)
        6931: "Overseas",   # Pathum Nissanka (Sri Lanka)
        4260: "Overseas",   # Heinrich Klaasen (South Africa)
        6280: "Overseas",   # Harry Brook (England)
        768: "Overseas",    # Reeza Hendricks (South Africa)
        4722: "Overseas",   # Rashid Khan (Afghanistan)
        2490: "Overseas",   # Mohammad Nawaz (Pakistan)
        4101: "Overseas",   # Mitchell Santner (New Zealand)
        4900: "Overseas",   # Wanindu Hasaranga (Sri Lanka)
        1758: "Overseas",   # Patrick Cummins (Australia)
        830: "Overseas",    # Josh Hazlewood (Australia)
        2382: "Overseas",   # Jamie Overton (England)
    }

    bat_df["nationality"] = bat_df["player_id"].map(overseas_override).fillna(bat_df["nationality"])
    bowl_df["nationality"] = bowl_df["player_id"].map(overseas_override).fillna(bowl_df["nationality"])

    # Data-grounded RR Squad Fit Notes (strictly empirical, no reputation bias)
    rr_fit_batters = {
        50284: "Has RR in his team history (through IPL 2026). High-leverage strike rate 212.4 vs baseline 223.1; positive resilience (+0.139 RV/ball) arises after context and opponent adjustments; ranks #1 in early and squeeze leverage; 75.8% P(Top-10).",
        7216: "Indian powerplay and middle-overs batter (Pressure SR 178.2, 74.1% match hit rate); +0.281 RV/ball resilience vs low-leverage baseline; 72.0% P(Top-10).",
        6281: "Indian LH top-order batter (Pressure SR 176.8 across 509 pressure balls); converts high-leverage powerplay opportunities (+0.382 posterior RV/ball); 68.2% P(Top-10).",
        3647: "Middle-overs anchor under chase and collapse pressure (Pressure SR 167.8); +0.233 RV/ball resilience vs baseline; 68.1% match hit rate under pressure; 69.0% P(Top-10).",
        6396: "Death-overs boundary hitter (Pressure SR 186.3 across 333 pressure balls); +0.172 RV/ball resilience; 70.3% match hit rate under pressure; 59.2% P(Top-10).",
        6931: "Top-order anchor (Pressure SR 143.8 across 567 pressure balls); highest match hit rate in pool (80.9%); +0.215 RV/ball resilience; 48.4% P(Top-10).",
        768: "Top-order batter (Pressure SR 171.5); +0.229 RV/ball resilience vs baseline; ranks top-15 across 3 alternate pressure definitions; 50.0% P(Top-10).",
        1699: "Highest volume exposure (594 pressure balls, 2,659 total balls faced); maintains +0.263 posterior RV/ball level and 174.0 Pressure SR; 24.4% P(Top-10).",
        6280: "Middle-to-death overs hitter (Pressure SR 156.5 across 162 pressure balls); +0.150 RV/ball resilience; ranks top-15 across 3 alternate pressure definitions; 47.4% P(Top-10).",
        4260: "Death-overs boundary hitter (Pressure SR 184.4 across 317 pressure balls); +0.101 RV/ball resilience; ranks top-15 across 2 alternate pressure definitions; 38.2% P(Top-10)."
    }

    rr_fit_bowlers = {
        781: "Defensive spin restrictor (Pressure Economy 7.02 across 416 legal pressure balls); ranks top-15 across 4 alternate pressure definitions; +0.204 posterior RV/ball; 81.8% P(Top-10).",
        4722: "High-volume leg-spin choke bowler (Pressure Econ 7.47 across 604 legal pressure balls); +0.189 posterior RV/ball; 62.0% match hit rate under pressure; 49.8% P(Top-10).",
        6256: "Mystery spin restrictor in middle and death overs (Pressure Econ 7.81 across 520 legal pressure balls); +0.181 posterior RV/ball; 49.8% P(Top-10).",
        2490: "Defensive left-arm orthodox spinner (Pressure Econ 7.07 across 298 legal pressure balls); +0.171 posterior RV/ball; ranks top-15 across 3 alternate pressure definitions; 58.6% P(Top-10).",
        4219: "Left-arm spin bowler with powerplay and middle-overs control (Pressure Econ 7.10 across 289 legal pressure balls); +0.128 posterior RV/ball and +0.107 resilience; 35.2% P(Top-10).",
        2382: "Pace-bowling all-rounder (Pressure Econ 6.26 across 68 legal pressure balls); +0.103 posterior RV/ball; 47.2% P(Top-10). Low-sample pick (68 legal pressure balls).",
        1612: "Has RR in his team history (through IPL 2026). Conceded 8.82 economy, offset by 47 wickets in 602 legal pressure balls at 12.81 balls/wicket (+0.27 runs/ball dismissal value credit); +0.120 posterior RV/ball; 22.8% P(Top-10).",
        4101: "Disciplined finger spinner (Pressure Econ 6.94 across 370 legal pressure balls); +0.127 posterior RV/ball; 65.3% match hit rate under pressure; 30.4% P(Top-10).",
        27960: "Pace bowler (Pressure Econ 7.28 across 108 legal pressure balls, 469 legal balls across 21 matches); +0.154 resilience; 38.0% P(Top-10). Low-sample pick (108 legal pressure balls).",
        4900: "Has RR in his team history (through IPL 2026). Attacking leg-spinner (Pressure Econ 8.61 across 455 legal pressure balls); +0.152 posterior RV/ball; 24.6% P(Top-10)."
    }

    bat_df["rr_auction_fit"] = bat_df["player_id"].map(rr_fit_batters).fillna("High-leverage situational batter.")
    bowl_df["rr_auction_fit"] = bowl_df["player_id"].map(rr_fit_bowlers).fillna("High-leverage situational bowler.")

    # Primary metrics
    bat_df["primary_metric_overall"] = bat_df["raw_sr"].round(1).astype(str) + " SR"
    bat_df["primary_metric_pressure"] = bat_df["pressure_sr"].round(1).astype(str) + " SR"

    bowl_df["primary_metric_overall"] = bowl_df["raw_econ"].round(2).astype(str) + " Econ"
    bowl_df["primary_metric_pressure"] = bowl_df["pressure_econ"].round(2).astype(str) + " Econ"

    # Ensure total balls and pressure balls columns are legal balls for bowlers
    bowl_df["n_balls"] = bowl_df["n_legal"]
    bowl_df["n_pressure_125"] = bowl_df["pressure_legal"]

    # Merge into single rankings dataframe
    rankings_final = pd.concat([bat_df, bowl_df], axis=0).reset_index(drop=True)

    # Column ordering
    cols = [
        "rank", "role", "player_id", "player_name", "display_name", "nationality",
        "teams", "is_rr", "composite_score", "score_ci_lower", "score_ci_upper",
        "p_top10", "p_top5", "rank_ci_lower", "rank_ci_upper", "n_alt_top15",
        "post_press", "post_base", "resilience", "hit_rate_shrunk",
        "n_balls", "n_pressure_125", "n_matches", "pct_ipl",
        "primary_metric_overall", "primary_metric_pressure", "rr_auction_fit"
    ]
    rankings_final = rankings_final[cols].copy()
    rankings_final.rename(columns={"n_balls": "total_balls_or_legal", "n_pressure_125": "pressure_balls", "is_rr": "is_rr_squad"}, inplace=True)

    # Round numeric columns for clean reporting
    round_cols = ["composite_score", "score_ci_lower", "score_ci_upper", "post_press", "post_base", "resilience"]
    for c in round_cols:
        rankings_final[c] = rankings_final[c].round(3)
    rankings_final["hit_rate_shrunk"] = (rankings_final["hit_rate_shrunk"] * 100).round(1).astype(str) + "%"
    rankings_final["p_top10"] = (rankings_final["p_top10"] * 100).round(1).astype(str) + "%"
    rankings_final["p_top5"] = (rankings_final["p_top5"] * 100).round(1).astype(str) + "%"
    rankings_final["pct_ipl"] = rankings_final["pct_ipl"].round(1).astype(str) + "%"
    rankings_final["rank_ci"] = "[" + rankings_final["rank_ci_lower"].astype(int).astype(str) + ", " + rankings_final["rank_ci_upper"].astype(int).astype(str) + "]"
    rankings_final["score_ci"] = "[" + rankings_final["score_ci_lower"].astype(str) + ", " + rankings_final["score_ci_upper"].astype(str) + "]"

    # Save rankings_final.csv in outputs and root
    out_csv = os.path.join(out_dir, "rankings_final.csv")
    rankings_final.to_csv(out_csv, index=False)
    shutil.copyfile(out_csv, "rankings_final.csv")
    print(f"Saved rankings_final.csv to {out_csv} and root directory.")

    # Generate dropdown picks file
    dropdown_path = os.path.join(out_dir, "dropdown_picks.txt")
    with open(dropdown_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SupeRR Selector Season 2 - Round 1: Pressure Players Dropdown Picks\n")
        f.write("=" * 70 + "\n\n")

        f.write("TOP 10 BATTERS UNDER PRESSURE (Select in Unstop Dropdowns Rank 1 -> 10):\n")
        f.write("-" * 70 + "\n")
        for _, r in bat_df.iterrows():
            rr_flag = " [Current RR Squad]" if r["is_rr"] else ""
            f.write(f"Rank {int(r['rank'])}: {r['display_name']} (Player ID: {r['player_id']}, {r['nationality']}){rr_flag}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 10 BOWLERS UNDER PRESSURE (Select in Unstop Dropdowns Rank 1 -> 10):\n")
        f.write("-" * 70 + "\n")
        for _, r in bowl_df.iterrows():
            rr_flag = " [Current RR Squad]" if r["is_rr"] else ""
            ambig = " *IMPORTANT: Select Afghanistan / GT player ID 4722, not Nepal ID 20172" if r["player_id"] == 4722 else ""
            f.write(f"Rank {int(r['rank'])}: {r['display_name']} (Player ID: {r['player_id']}, {r['nationality']}){rr_flag}{ambig}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DROPDOWN AMBIGUITY SAFETY NOTE:\n")
        f.write("There are duplicate names in the dataset. For Rank 2 Bowler 'Rashid Khan', ensure\n")
        f.write("selecting the Afghanistan/Gujarat Titans superstar (player_id 4722).\n")
        f.write("=" * 70 + "\n")

    shutil.copyfile(dropdown_path, "dropdown_picks.txt")
    print(f"Saved dropdown picks file to {dropdown_path} and root directory.")

    # Generate Top 10 Summary Graphic
    plot_top10_summary(bat_df, bowl_df, config)

    return rankings_final


def plot_top10_summary(bat_df, bowl_df, config):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Batters
    y = np.arange(len(bat_df))[::-1]
    axes[0].barh(y, bat_df["composite_score"], color="#1f77b4", alpha=0.85, edgecolor="black")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"#{int(r['rank'])} {r['display_name']} ({r['nationality']})" for _, r in bat_df.iterrows()], fontsize=10)
    for i, (_, r) in enumerate(bat_df.iterrows()):
        idx = len(bat_df) - 1 - i
        axes[0].text(r["composite_score"] + 0.05, idx, f"Score: {r['composite_score']:.2f} | SR: {r['pressure_sr']:.0f}", va="center", fontsize=9)
    axes[0].set_title("Top 10 Batters Under Pressure", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Composite Pressure Rating")
    axes[0].set_xlim(0, 3.2)
    axes[0].grid(True, alpha=0.3)

    # Bowlers
    y_bw = np.arange(len(bowl_df))[::-1]
    axes[1].barh(y_bw, bowl_df["composite_score"], color="#2ca02c", alpha=0.85, edgecolor="black")
    axes[1].set_yticks(y_bw)
    axes[1].set_yticklabels([f"#{int(r['rank'])} {r['display_name']} ({r['nationality']})" for _, r in bowl_df.iterrows()], fontsize=10)
    for i, (_, r) in enumerate(bowl_df.iterrows()):
        idx = len(bowl_df) - 1 - i
        axes[1].text(r["composite_score"] + 0.05, idx, f"Score: {r['composite_score']:.2f} | Econ: {r['pressure_econ']:.1f}", va="center", fontsize=9)
    axes[1].set_title("Top 10 Bowlers Under Pressure", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Composite Pressure Rating")
    axes[1].set_xlim(0, 3.2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(config["paths"]["figures_dir"], "top10_summary_bars.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Top-10 summary bar chart to {fig_path}")


if __name__ == "__main__":
    cfg = load_config()
    generate_final_outputs(cfg)
