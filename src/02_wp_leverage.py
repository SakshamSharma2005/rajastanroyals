"""
02_wp_leverage.py - Win Probability Models, Leverage Index (LI), Effective LI, and Bite Test.
Follows PLAN_SupeRR_Selector_R1_Pressure_Players.md §4.2, §7.3.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_wp_models(df, config):
    print("--- Training Win Probability Models ---")
    # Non-DLS, non-null bat_won
    train_mask = (~df["is_dls"]) & (df["bat_won"].notna())
    
    # 1. Chase model (Innings 2)
    chase_mask = train_mask & (df["innings"] == 2)
    chase_df = df[chase_mask].copy()
    chase_feats = ["balls_left", "wk_in_hand", "runs_needed", "rrr", "crr", "rrr_minus_crr", "runs_before", "is_ipl"]
    X_chase = chase_df[chase_feats].copy()
    X_chase["is_ipl"] = X_chase["is_ipl"].astype(int)
    y_chase = chase_df["bat_won"].astype(int)
    groups_chase = chase_df["match_id"]

    chase_cfg = config["wp_model"]["chase"]
    gkf = GroupKFold(n_splits=chase_cfg.get("cv_splits", 5))
    oof_chase = np.zeros(len(chase_df))

    print("Evaluating Chase WP model with 5-fold GroupKFold...")
    for train_idx, val_idx in gkf.split(X_chase, y_chase, groups_chase):
        clf = HistGradientBoostingClassifier(
            max_depth=chase_cfg["max_depth"],
            learning_rate=chase_cfg["learning_rate"],
            max_iter=chase_cfg["max_iter"],
            min_samples_leaf=chase_cfg["min_samples_leaf"],
            l2_regularization=chase_cfg["l2_regularization"],
            random_state=config["seed"]
        )
        clf.fit(X_chase.iloc[train_idx], y_chase.iloc[train_idx])
        oof_chase[val_idx] = clf.predict_proba(X_chase.iloc[val_idx])[:, 1]

    auc_chase = roc_auc_score(y_chase, oof_chase)
    loss_chase = log_loss(y_chase, oof_chase)
    brier_chase = brier_score_loss(y_chase, oof_chase)
    print(f"Chase WP OOF -> AUC: {auc_chase:.4f}, LogLoss: {loss_chase:.4f}, Brier: {brier_chase:.4f}")

    # AUC by balls_left bucket
    chase_bucket_auc = {}
    for b_min, b_max, name in [
        (0, 30, "0-30 balls"),
        (31, 60, "31-60 balls"),
        (61, 90, "61-90 balls"),
        (91, 120, "91-120 balls")
    ]:
        b_mask = (chase_df["balls_left"] >= b_min) & (chase_df["balls_left"] <= b_max)
        sub_auc = roc_auc_score(y_chase[b_mask], oof_chase[b_mask])
        chase_bucket_auc[name] = float(sub_auc)
        print(f"  Chase {name} AUC: {sub_auc:.4f}")

    # Fit final Chase model on all non-DLS data
    final_clf_chase = HistGradientBoostingClassifier(
        max_depth=chase_cfg["max_depth"],
        learning_rate=chase_cfg["learning_rate"],
        max_iter=chase_cfg["max_iter"],
        min_samples_leaf=chase_cfg["min_samples_leaf"],
        l2_regularization=chase_cfg["l2_regularization"],
        random_state=config["seed"]
    )
    final_clf_chase.fit(X_chase, y_chase)

    # 2. Innings 1 model
    inn1_mask = train_mask & (df["innings"] == 1)
    inn1_df = df[inn1_mask].copy()
    inn1_feats = ["runs_before", "wk_in_hand", "balls_left", "crr", "is_ipl"]
    X_inn1 = inn1_df[inn1_feats].copy()
    X_inn1["is_ipl"] = X_inn1["is_ipl"].astype(int)
    y_inn1 = inn1_df["bat_won"].astype(int)
    groups_inn1 = inn1_df["match_id"]

    inn1_cfg = config["wp_model"]["inn1"]
    gkf_inn1 = GroupKFold(n_splits=inn1_cfg.get("cv_splits", 5))
    oof_inn1 = np.zeros(len(inn1_df))

    print("Evaluating Innings 1 WP model with 5-fold GroupKFold...")
    for train_idx, val_idx in gkf_inn1.split(X_inn1, y_inn1, groups_inn1):
        clf = HistGradientBoostingClassifier(
            max_depth=inn1_cfg["max_depth"],
            learning_rate=inn1_cfg["learning_rate"],
            max_iter=inn1_cfg["max_iter"],
            min_samples_leaf=inn1_cfg["min_samples_leaf"],
            l2_regularization=inn1_cfg["l2_regularization"],
            random_state=config["seed"]
        )
        clf.fit(X_inn1.iloc[train_idx], y_inn1.iloc[train_idx])
        oof_inn1[val_idx] = clf.predict_proba(X_inn1.iloc[val_idx])[:, 1]

    auc_inn1 = roc_auc_score(y_inn1, oof_inn1)
    loss_inn1 = log_loss(y_inn1, oof_inn1)
    brier_inn1 = brier_score_loss(y_inn1, oof_inn1)
    print(f"Inn1 WP OOF -> AUC: {auc_inn1:.4f}, LogLoss: {loss_inn1:.4f}, Brier: {brier_inn1:.4f}")

    inn1_bucket_auc = {}
    for b_min, b_max, name in [
        (0, 30, "0-30 balls"),
        (31, 60, "31-60 balls"),
        (61, 90, "61-90 balls"),
        (91, 120, "91-120 balls")
    ]:
        b_mask = (inn1_df["balls_left"] >= b_min) & (inn1_df["balls_left"] <= b_max)
        sub_auc = roc_auc_score(y_inn1[b_mask], oof_inn1[b_mask])
        inn1_bucket_auc[name] = float(sub_auc)
        print(f"  Inn1 {name} AUC: {sub_auc:.4f}")

    final_clf_inn1 = HistGradientBoostingClassifier(
        max_depth=inn1_cfg["max_depth"],
        learning_rate=inn1_cfg["learning_rate"],
        max_iter=inn1_cfg["max_iter"],
        min_samples_leaf=inn1_cfg["min_samples_leaf"],
        l2_regularization=inn1_cfg["l2_regularization"],
        random_state=config["seed"]
    )
    final_clf_inn1.fit(X_inn1, y_inn1)

    metrics = {
        "chase": {
            "auc": float(auc_chase),
            "log_loss": float(loss_chase),
            "brier": float(brier_chase),
            "bucket_auc": chase_bucket_auc
        },
        "inn1": {
            "auc": float(auc_inn1),
            "log_loss": float(loss_inn1),
            "brier": float(brier_inn1),
            "bucket_auc": inn1_bucket_auc
        }
    }

    # Plot calibration curves
    plot_wp_calibration(y_chase, oof_chase, y_inn1, oof_inn1, config)

    return final_clf_chase, final_clf_inn1, metrics, (chase_feats, inn1_feats)


def plot_wp_calibration(y_chase, oof_chase, y_inn1, oof_inn1, config):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chase
    prob_true_c, prob_pred_c = calibration_curve(y_chase, oof_chase, n_bins=10)
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(prob_pred_c, prob_true_c, "s-", color="#1f77b4", label="Chase Model (OOF)")
    axes[0].set_title(f"Chase WP Calibration (AUC: {roc_auc_score(y_chase, oof_chase):.3f})")
    axes[0].set_xlabel("Mean Predicted Win Probability")
    axes[0].set_ylabel("Empirical Win Fraction")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Innings 1
    prob_true_1, prob_pred_1 = calibration_curve(y_inn1, oof_inn1, n_bins=10)
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[1].plot(prob_pred_1, prob_true_1, "o-", color="#ff7f0e", label="Inn 1 Model (OOF)")
    axes[1].set_title(f"1st Innings WP Calibration (AUC: {roc_auc_score(y_inn1, oof_inn1):.3f})")
    axes[1].set_xlabel("Mean Predicted Win Probability")
    axes[1].set_ylabel("Empirical Win Fraction")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(config["paths"]["figures_dir"], "wp_calibration.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved WP calibration plot to {fig_path}")


def compute_leverage_index(df, clf_chase, clf_inn1, feats_chase, feats_inn1, config):
    print("--- Computing Leverage Index (LI) ---")
    
    # Outcome classification
    def get_outcome(row):
        et = row["extras_type"]
        if et == "wide":
            return "wide"
        elif et == "Overstep":
            return "no_ball"
        elif row["is_wicket"] == 1:
            return "wicket"
        else:
            r = int(row["runs"])
            if r == 0:
                return "0"
            elif r == 1:
                return "1"
            elif r == 2:
                return "2"
            elif r == 3:
                return "3"
            elif r == 4:
                return "4"
            else:
                return "6"

    df["outcome_type"] = df.apply(get_outcome, axis=1)

    # Wicket bucket
    wb = np.where(df["wk_before"] <= 1, "0-1", np.where(df["wk_before"] <= 3, "2-3", "4+"))
    df["wk_bucket"] = wb

    # Frequency table: P(outcome | innings, phase, wk_bucket)
    p_table = df.groupby(["innings", "phase", "wk_bucket", "outcome_type"]).size().unstack(fill_value=0)
    p_table = p_table.div(p_table.sum(axis=1), axis=0)

    # 9 outcomes specification
    outcomes = [
        # name, runs_added, legal_consumed, wicket_lost
        ("0", 0.0, 1, 0),
        ("1", 1.0, 1, 0),
        ("2", 2.0, 1, 0),
        ("3", 3.0, 1, 0),
        ("4", 4.0, 1, 0),
        ("6", 6.0, 1, 0),
        ("wicket", 0.0, 1, 1),
        ("wide", 1.0, 0, 0),
        ("no_ball", 2.0, 0, 0),
    ]

    # Evaluate current WP
    # Chase
    chase_mask = df["innings"] == 2
    inn1_mask = df["innings"] == 1

    wp_current = np.zeros(len(df), dtype=float)

    # Current Chase WP
    Xc = df.loc[chase_mask, feats_chase].copy()
    Xc["is_ipl"] = Xc["is_ipl"].astype(int)
    wp_current[chase_mask] = clf_chase.predict_proba(Xc)[:, 1]

    # Current Inn1 WP
    Xi = df.loc[inn1_mask, feats_inn1].copy()
    Xi["is_ipl"] = Xi["is_ipl"].astype(int)
    wp_current[inn1_mask] = clf_inn1.predict_proba(Xi)[:, 1]

    df["wp_before"] = wp_current

    # Compute expected absolute swing for each ball
    # Join outcome probabilities to df
    df_keys = df[["innings", "phase", "wk_bucket"]].set_index(["innings", "phase", "wk_bucket"])
    probs_df = p_table.loc[df_keys.index].values  # shape (N, 9)
    col_names = p_table.columns.tolist()

    lev = np.zeros(len(df), dtype=float)

    # Compute next WP for each outcome
    print("Evaluating hypothetical state transitions for leverage index...")
    for out_name, r_add, l_cons, w_lost in outcomes:
        if out_name not in col_names:
            continue
        p_k = probs_df[:, col_names.index(out_name)]

        wp_next = np.zeros(len(df), dtype=float)

        # 1. Chase next WP
        if chase_mask.sum() > 0:
            c_sub = df.loc[chase_mask]
            next_balls_left = np.clip(c_sub["balls_left"].values - l_cons, 0, 120)
            next_runs_before = c_sub["runs_before"].values + r_add
            next_wk_in_hand = np.clip(c_sub["wk_in_hand"].values - w_lost, 0, 10)
            next_target = c_sub["target"].values
            next_runs_needed = np.maximum(0.0, next_target - next_runs_before)
            
            overs_left = np.maximum(next_balls_left / 6.0, 1.0 / 6.0)
            next_rrr = np.where(next_balls_left > 0, next_runs_needed / overs_left, 99.0)
            overs_bowled = np.maximum((120 - next_balls_left) / 6.0, 1.0 / 6.0)
            next_crr = next_runs_before / overs_bowled
            next_rrr_minus_crr = next_rrr - next_crr

            # Terminal states
            term_win = next_runs_needed <= 0
            term_loss = (next_wk_in_hand <= 0) | ((next_balls_left == 0) & (next_runs_needed > 0))
            non_term = (~term_win) & (~term_loss)

            wp_next_c = np.zeros(len(c_sub), dtype=float)
            wp_next_c[term_win] = 1.0
            wp_next_c[term_loss] = 0.0

            if non_term.sum() > 0:
                X_next_c = pd.DataFrame({
                    "balls_left": next_balls_left[non_term],
                    "wk_in_hand": next_wk_in_hand[non_term],
                    "runs_needed": next_runs_needed[non_term],
                    "rrr": next_rrr[non_term],
                    "crr": next_crr[non_term],
                    "rrr_minus_crr": next_rrr_minus_crr[non_term],
                    "runs_before": next_runs_before[non_term],
                    "is_ipl": c_sub["is_ipl"].values[non_term].astype(int)
                })
                wp_next_c[non_term] = clf_chase.predict_proba(X_next_c)[:, 1]
            wp_next[chase_mask] = wp_next_c

        # 2. Inn1 next WP
        if inn1_mask.sum() > 0:
            i_sub = df.loc[inn1_mask]
            next_balls_left = np.clip(i_sub["balls_left"].values - l_cons, 0, 120)
            next_runs_before = i_sub["runs_before"].values + r_add
            next_wk_in_hand = np.clip(i_sub["wk_in_hand"].values - w_lost, 0, 10)
            overs_bowled = np.maximum((120 - next_balls_left) / 6.0, 1.0 / 6.0)
            next_crr = next_runs_before / overs_bowled

            term_allout = next_wk_in_hand <= 0
            wp_next_i = np.zeros(len(i_sub), dtype=float)

            X_next_i = pd.DataFrame({
                "runs_before": next_runs_before,
                "wk_in_hand": next_wk_in_hand,
                "balls_left": next_balls_left,
                "crr": next_crr,
                "is_ipl": i_sub["is_ipl"].values.astype(int)
            })
            # For allout, zero balls left
            X_next_i.loc[term_allout, "balls_left"] = 0
            wp_next_i = clf_inn1.predict_proba(X_next_i)[:, 1]
            wp_next[inn1_mask] = wp_next_i

        lev += p_k * np.abs(wp_next - wp_current)

    mean_lev = np.mean(lev)
    df["li"] = lev / mean_lev
    print(f"Raw LI computed -> Mean: {df['li'].mean():.3f}, Median: {df['li'].median():.3f}, Max: {df['li'].max():.3f}")

    # Phase reliability weight rho(innings, phase)
    # AUC informs reliability: chase late AUC is ~0.954, so denominator is (0.954 - 0.5) = 0.454
    # Inn1 AUC early is ~0.588, late is ~0.734
    rho_map = {
        (1, "PP"): 0.35,
        (1, "M1"): 0.45,
        (1, "M2"): 0.55,
        (1, "D"): 0.65,
        (2, "PP"): 0.70,
        (2, "M1"): 0.85,
        (2, "M2"): 0.95,
        (2, "D"): 1.00,
    }
    df["rho"] = [rho_map.get((inn, ph), 0.7) for inn, ph in zip(df["innings"], df["phase"])]

    # Innings 1 minimum decision window:
    # Pressure before ball 25 counts only if team has already lost >= 3 wickets (early collapse)
    inn1_early_discount = np.where(
        (df["innings"] == 1) & (df["legal_before"] < 24) & (df["wk_before"] < 3),
        0.5,
        1.0
    )

    df["li_eff"] = df["li"] * df["rho"] * inn1_early_discount
    # Re-normalize li_eff so mean = 1.0
    df["li_eff"] = df["li_eff"] / df["li_eff"].mean()
    print(f"Effective LI (li_eff) -> Mean: {df['li_eff'].mean():.3f}, Median: {df['li_eff'].median():.3f}, Max: {df['li_eff'].max():.3f}")

    # Compute actual WP after ball
    # Next actual state
    grp = df.groupby(["match_id", "innings"])
    # Shift wp_before from the next ball within same innings/match
    df["wp_after"] = grp["wp_before"].shift(-1)
    
    # Handle the last delivery of innings
    # For Innings 2: if chase ended, final outcome is bat_won
    last_ball_c = chase_mask & df["wp_after"].isna()
    df.loc[last_ball_c, "wp_after"] = df.loc[last_ball_c, "bat_won"]
    
    # For Innings 1: use model prediction at 120 balls / final total
    last_ball_i = inn1_mask & df["wp_after"].isna()
    df.loc[last_ball_i, "wp_after"] = df.loc[last_ball_i, "wp_before"]

    df["wpa"] = df["wp_after"] - df["wp_before"]

    return df


def run_bite_test(df, config):
    print("--- Running Bite Test (§4.2) ---")
    bins = [0, 0.75, 1.25, 2.0, 3.0, np.inf]
    labels = ["<0.75", "0.75-1.25", "1.25-2.0", "2.0-3.0", "3.0+"]
    df["li_bin"] = pd.cut(df["li"], bins=bins, labels=labels, right=False)

    # Chase bite test
    chase = df[df["innings"] == 2].copy()
    chase_bite = chase.groupby("li_bin", observed=False).agg(
        balls=("runs", "count"),
        wkt_per_100=("is_wicket", lambda x: np.mean(x) * 100),
        bowl_wkt_per_100=("is_bowler_wicket", lambda x: np.mean(x) * 100),
        dot_pct=("runs", lambda x: np.mean(x == 0) * 100),
        strike_rate=("runs", lambda x: np.sum(x) / np.maximum(np.sum(chase.loc[x.index, "legal"]), 1) * 100)
    ).reset_index()
    chase_bite["innings"] = "Chase (2nd Inn)"

    # Innings 1 bite test
    inn1 = df[df["innings"] == 1].copy()
    inn1_bite = inn1.groupby("li_bin", observed=False).agg(
        balls=("runs", "count"),
        wkt_per_100=("is_wicket", lambda x: np.mean(x) * 100),
        bowl_wkt_per_100=("is_bowler_wicket", lambda x: np.mean(x) * 100),
        dot_pct=("runs", lambda x: np.mean(x == 0) * 100),
        strike_rate=("runs", lambda x: np.sum(x) / np.maximum(np.sum(inn1.loc[x.index, "legal"]), 1) * 100)
    ).reset_index()
    inn1_bite["innings"] = "1st Innings"

    bite_all = pd.concat([chase_bite, inn1_bite], axis=0)
    bite_csv_path = os.path.join(config["paths"]["output_dir"], "bite_test.csv")
    bite_all.to_csv(bite_csv_path, index=False)
    print(f"Saved bite test to {bite_csv_path}")
    print(chase_bite)
    print(inn1_bite[["li_bin", "wkt_per_100", "dot_pct", "strike_rate"]])

    # Plot Bite Test Figure
    plot_bite_test(chase_bite, inn1_bite, config)


def plot_bite_test(chase_bite, inn1_bite, config):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Chase Wickets and Strike Rate
    ax1 = axes[0]
    x = np.arange(len(chase_bite))
    width = 0.35
    ax1.bar(x - width/2, chase_bite["wkt_per_100"], width, label="Wickets / 100 balls", color="#d62728", alpha=0.85)
    ax1.bar(x + width/2, chase_bite["bowl_wkt_per_100"], width, label="Bowler Wkts / 100 balls", color="#ff7f0e", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(chase_bite["li_bin"])
    ax1.set_title("Chase (2nd Inn): Wicket Hazard Escalates Under Pressure")
    ax1.set_xlabel("Leverage Index (LI) Bucket")
    ax1.set_ylabel("Wickets per 100 Deliveries")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1st Innings Dot % and Chase Dot %
    ax2 = axes[1]
    ax2.plot(chase_bite["li_bin"], chase_bite["dot_pct"], "o-", color="#1f77b4", linewidth=2.5, label="Chase Dot %")
    ax2.plot(inn1_bite["li_bin"], inn1_bite["dot_pct"], "s--", color="#2ca02c", linewidth=2.5, label="1st Inn Dot %")
    ax2.set_title("Dot Ball Pressure: Batters Pinned or Squeezed")
    ax2.set_xlabel("Leverage Index (LI) Bucket")
    ax2.set_ylabel("Dot Ball Percentage (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(config["paths"]["figures_dir"], "bite_test.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved bite test figure to {fig_path}")


def plot_li_distribution(df, config):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # LI by over
    over_li = df.groupby(["overid", "innings"])["li_eff"].mean().unstack()
    axes[0].plot(over_li.index, over_li[1], "o-", color="#2ca02c", label="1st Innings")
    axes[0].plot(over_li.index, over_li[2], "s-", color="#1f77b4", label="Chase (2nd Innings)")
    axes[0].set_title("Mean Effective Leverage (Pressure) by Over")
    axes[0].set_xlabel("Over (1-20)")
    axes[0].set_ylabel("Mean Effective LI")
    axes[0].set_xticks(range(1, 21))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LI distribution density
    sns.kdeplot(df.loc[df["innings"] == 1, "li_eff"], ax=axes[1], color="#2ca02c", label="1st Innings", clip=(0, 6))
    sns.kdeplot(df.loc[df["innings"] == 2, "li_eff"], ax=axes[1], color="#1f77b4", label="Chase", clip=(0, 6))
    axes[1].axvline(1.5, color="red", linestyle="--", label="Pressure Cutoff (LI=1.5)")
    axes[1].set_title("Distribution of Effective Leverage Index")
    axes[1].set_xlabel("Effective LI")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(config["paths"]["figures_dir"], "li_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved LI distribution plot to {fig_path}")


if __name__ == "__main__":
    cfg = load_config()
    clean_path = os.path.join(cfg["paths"]["output_dir"], "clean_deliveries.parquet")
    df = pd.read_parquet(clean_path)

    clf_chase, clf_inn1, metrics, (feats_chase, feats_inn1) = train_wp_models(df, cfg)
    
    with open(os.path.join(cfg["paths"]["output_dir"], "wp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    df = compute_leverage_index(df, clf_chase, clf_inn1, feats_chase, feats_inn1, cfg)
    run_bite_test(df, cfg)
    plot_li_distribution(df, cfg)

    out_path = os.path.join(cfg["paths"]["output_dir"], "ball_state_wp.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Step 2 complete! Saved enriched deliveries to {out_path}")
