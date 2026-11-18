# SupeRR Selector Season 2 · Round 1: Pressure Players Pipeline

### Most Reliable Players Under Pressure in T20 Cricket (IPL & T20I 2022–2026)

This repository contains the complete, reproducible, production-grade sports data science pipeline implementing `PLAN_SupeRR_Selector_R1_Pressure_Players.md` end-to-end. It values player performance under pressure using a win-probability Leverage Index (LI), dynamic-programming run-expectancy tables $V(b, w)$, joint Ridge regression for opposition/venue adjustments, Empirical Bayes shrinkage, and match-by-match consistency metrics.

Every single ranking and confidence interval traces strictly to computed data formulas; **zero players were hand-picked**.

---

## 1. Final Official Rankings

### Top 10 Batters Under Pressure (Rank 1 → 10)

| Rank | Player Name | Player ID | Nationality | Primary Teams | Score | 90% Cluster Bootstrap CI | P(Top-10) | Pressure Strike Rate | Strategic RR Fit / Rationale |
|:---:|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **1** | **Vaibhav Sooryavanshi** | 50284 | Indian | India, India A, Rajasthan Royals | **2.397** | [1.012, 3.782] | 76.6% | **212.4 SR** | Current RR squad asset. High-leverage strike rate 212.4 vs baseline 223.1; positive resilience (+0.17 RV/ball); ranks #1 in early/squeeze leverage. |
| **2** | **Prabhsimran Singh** | 7216 | Indian | India, India A, Punjab Kings | **2.158** | [1.216, 3.100] | 72.8% | **178.2 SR** | Aggressive Indian PP & middle-overs batter (Pressure SR 178.2, 74.1% match hit rate); +0.28 RV/ball resilience vs low-leverage baseline. |
| **3** | **Abhishek Sharma** | 6281 | Indian | India, India A, Sunrisers Hyderabad | **2.126** | [1.238, 3.013] | 69.4% | **176.8 SR** | High-impact Indian LH top-order batter (Pressure SR 176.8); strong volume (509 pressure balls); converts high-leverage PP opportunities (+0.38 posterior RV/ball). |
| **4** | **Shreyas Iyer** | 3647 | Indian | India, Kolkata Knight Riders, PBKS | **2.015** | [1.357, 2.672] | 70.8% | **167.8 SR** | Dependable middle-overs batter under chase/collapse pressure (Pressure SR 167.8); +0.24 RV/ball resilience vs baseline; 68.1% match hit rate under pressure. |
| **5** | **Tim David** | 6396 | Overseas | Australia, Mumbai Indians, RCB | **1.986** | [0.998, 2.974] | 60.2% | **186.3 SR** | Death-overs finisher (Pressure SR 186.3 across 333 pressure balls); +0.19 RV/ball resilience; excels in high required-run-rate chase scenarios. |
| **6** | **Pathum Nissanka** | 6931 | Overseas | Delhi Capitals, Sri Lanka | **1.871** | [1.203, 2.538] | 50.0% | **143.8 SR** | High-tempo overseas top-order batter (Pressure SR 143.8 across 567 pressure balls); strong consistency (80.9% match hit rate under pressure); +0.22 RV/ball resilience. |
| **7** | **Reeza Hendricks** | 768 | Overseas | South Africa | **1.704** | [0.940, 2.468] | 50.4% | **171.5 SR** | Experienced overseas top-order batter (Pressure SR 171.5); +0.23 RV/ball resilience; ranks top-15 across 3 alternate pressure definitions. |
| **8** | **Suryakumar Yadav** | 1699 | Indian | India, Mumbai Indians | **1.613** | [0.967, 2.260] | 27.4% | **174.0 SR** | High-volume middle-overs batter (594 pressure balls, Pressure SR 174.0); +0.26 posterior RV/ball level; dependable situational performer. |
| **9** | **Harry Brook** | 6280 | Overseas | Delhi Capitals, England, SRH | **1.415** | [0.345, 2.486] | 47.6% | **156.5 SR** | Middle-to-death overs hitter (Pressure SR 156.5 across 162 pressure balls); +0.16 RV/ball resilience; ranks top-15 across 4 alternate pressure definitions. |
| **10** | **Heinrich Klaasen** | 4260 | Overseas | South Africa, Sunrisers Hyderabad | **1.392** | [0.571, 2.213] | 40.2% | **184.4 SR** | High-impact middle-and-death finisher (Pressure SR 184.4 across 317 pressure balls); +0.11 RV/ball resilience; ranks top-15 across 3 alternate pressure definitions. |

---

### Top 10 Bowlers Under Pressure (Rank 1 → 10)

| Rank | Player Name | Player ID | Nationality | Primary Teams | Score | 90% Cluster Bootstrap CI | P(Top-10) | Pressure Economy | Strategic RR Fit / Rationale |
|:---:|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **1** | **Sunil Narine** | 781 | Overseas | Kolkata Knight Riders | **2.433** | [1.667, 3.199] | 85.8% | **7.02 Econ** | Defensive spin restrictor (Pressure Economy 7.02 across 420 pressure balls); ranks top-15 across all 5 alternate pressure definitions; +0.22 posterior RV/ball. |
| **2** | **Rashid Khan (Afghanistan)** | 4722 | Overseas | Afghanistan, Gujarat Titans | **2.314** | [1.492, 3.136] | 57.4% | **7.47 Econ** | High-volume leg-spin choke bowler (Pressure Econ 7.47 across 611 pressure balls); +0.20 posterior RV/ball; 62.0% match hit rate under pressure. (*ID 4722, not Nepal ID 20172*). |
| **3** | **Varun Chakravarthy** | 6256 | Indian | India, Kolkata Knight Riders | **1.979** | [1.236, 2.723] | 59.0% | **7.81 Econ** | Mystery spin restrictor in middle and death overs (Pressure Econ 7.81 across 532 pressure balls); +0.19 posterior RV/ball. |
| **4** | **Mohammad Nawaz** | 2490 | Overseas | Pakistan | **1.868** | [0.821, 2.914] | 63.2% | **7.07 Econ** | Defensive left-arm orthodox spinner (Pressure Econ 7.07 across 301 pressure balls); +0.18 posterior RV/ball; ranks top-15 across 4 alternate pressure definitions. |
| **5** | **Krunal Pandya** | 4219 | Indian | Lucknow Super Giants, RCB | **1.668** | [0.814, 2.523] | 42.2% | **7.10 Econ** | Left-arm spin bowler with powerplay and middle-overs control (Pressure Econ 7.10 across 291 pressure balls); +0.14 posterior RV/ball and +0.11 resilience. |
| **6** | **Yuzvendra Chahal** | 1612 | Indian | India, Punjab Kings, Rajasthan Royals | **1.644** | [0.638, 2.650] | 29.0% | **8.82 Econ** | High-volume leg-spinner (Pressure Econ 8.82 across 626 pressure balls); strong wicket hazard; +0.13 posterior RV/ball and +0.15 resilience. |
| **7** | **Mitchell Santner** | 4101 | Overseas | Chennai Super Kings, NZ, MI | **1.538** | [0.788, 2.289] | 35.0% | **6.94 Econ** | Economical finger spinner (Pressure Econ 6.94 across 386 pressure balls); +0.13 posterior RV/ball; 65.3% match hit rate under pressure. |
| **8** | **Mayank Yadav** | 27960 | Indian | Lucknow Super Giants | **1.412** | [0.023, 2.801] | 40.2% | **7.28 Econ** | Express pace enforcer (Pressure Econ 7.28 across 110 pressure balls); +0.16 RV/ball resilience vs baseline; ranks top-15 across 4 alternate pressure definitions. |
| **9** | **Wanindu Hasaranga** | 4900 | Overseas | Sri Lanka, Sunrisers Hyderabad, RR | **1.390** | [0.690, 2.090] | 30.2% | **8.61 Econ** | Attacking leg-spinner (Pressure Econ 8.61 across 460 pressure balls); +0.16 posterior RV/ball; ranks top-15 across 4 alternate pressure definitions. |
| **10** | **Patrick Cummins** | 1758 | Overseas | Australia, Sunrisers Hyderabad | **1.372** | [0.461, 2.283] | 27.4% | **7.78 Econ** | Pace bowler across powerplay and death (Pressure Econ 7.78 across 266 pressure balls); +0.10 resilience; ranks top-15 across 3 alternate pressure definitions. |

---

## 2. Dropdown Pick Reference & Disambiguation Guide

When entering selections in the Unstop dropdown menus:

> [!IMPORTANT]
> **Rashid Khan Disambiguation:** Two distinct players named "Rashid Khan" exist in the dataset:
> - **Select:** `Rashid Khan` (Afghanistan / Gujarat Titans) with **`player_id: 4722`**.
> - **Do NOT Select:** `Rashid Khan` (Nepal) with `player_id: 20172`.
> 
> The text summary file is saved at `dropdown_picks.txt` and `outputs/dropdown_picks.txt`.

---

## 3. Data Integrity & Reconciled Counts

| Metric / Audit Parameter | Raw Total | Cleaning Treatment | Valid Analytical Sample |
|:---|:---:|:---|:---:|
| **Matches** | 883 | Excluded 6 Super Overs (38 balls) & 34 DLS matches (4,395 balls) | **833 valid matches** (362 IPL, 389 Full Member T20I, 52 Associate T20I, 30 A-Team) |
| **Deliveries** | 200,961 | Net of Super Overs & DLS | **196,528 valid balls** (86,668 IPL, 91,001 Full Member T20I, 11,886 Associate, 6,973 A-Team) |
| **Legal Ball Rate** | 200,961 | Only non-wides / non-noballs count toward 6 balls per over | **99.99%** of non-terminal overs have exactly 6 legal deliveries (29,964 of 29,968 overs) |
| **Team RV Balance** | 196,528 | DP tables include empirical extra rate (~0.05 runs/ball) | **-0.00037 runs/ball (IPL)** and **+0.00124 runs/ball (T20I)** (both $\ll 0.05$ tolerance) |
| **Ridge Regularization** | 1,600+ effects | Simultaneous RAPM Ridge ($\alpha = 1000.0$) | Bowler difficulty $[-0.211, +0.149]$ runs/ball; Venue effects $[-0.135, +0.184]$ runs/ball |

---

## 4. Repository Structure & Execution Instructions

```
F:/rr/
├── config.yaml                    # Master parameter log and configuration
├── README.md                      # Reproducibility manual & methodology overview
├── rankings_final.csv             # Official Top-10 Batters & Bowlers table with CIs & P(Top-10)
├── dropdown_picks.txt             # Plain-text Unstop dropdown pick instructions
├── methodology_report.pdf         # 13-page publication-grade PDF report
├── tests/
│   └── test_pipeline.py           # Automated Pytest suite (6/6 tests passing)
├── src/
│   ├── 01_load_clean.py           # Ingestion, cleaning, state reconstruction, tiering
│   ├── 02_wp_leverage.py          # Win probability models, LI, and Bite Test
│   ├── 03_run_value.py            # Dynamic programming run expectancy V(b, w) & RV
│   ├── 04_adjust.py               # 8-bin context centring & joint Ridge regression
│   ├── 05_player_ratings.py       # EB shrinkage, resilience, match consistency, composite
│   ├── 06_validation.py           # Split-half reliability, cluster bootstrap, sensitivity, LOCO
│   ├── 07_outputs.py              # Final formatting, RR fit mapping, dropdown export
│   └── generate_report.py         # ReportLab PDF report generation script
└── outputs/
    ├── clean_deliveries.parquet
    ├── ball_state_wp.parquet
    ├── ball_state_rv.parquet
    ├── ball_state_adj.parquet
    ├── rankings_batters_all.csv
    ├── rankings_bowlers_all.csv
    ├── rankings_final.csv
    ├── dropdown_picks.txt
    ├── methodology_report.pdf
    ├── bite_test.csv
    ├── reliability_results.csv
    ├── stability.csv
    ├── sensitivity_grid.csv
    ├── loco_results.csv
    ├── sanity_adjustments.csv
    └── figures/
        ├── wp_calibration.png
        ├── bite_test.png
        ├── li_distribution.png
        ├── pressure_vs_baseline.png
        ├── reliability_comparison.png
        ├── forest_plot_batters.png
        ├── forest_plot_bowlers.png
        ├── sensitivity_heatmap.png
        └── top10_summary_bars.png
```

### Reproducing the Pipeline from Scratch

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn reportlab pypdf pyyaml pytest
   ```

2. **Run Pipeline End-to-End:**
   ```bash
   python src/01_load_clean.py
   python src/02_wp_leverage.py
   python src/03_run_value.py
   python src/04_adjust.py
   python src/05_player_ratings.py
   python src/06_validation.py
   python src/07_outputs.py
   python src/generate_report.py
   ```

3. **Execute Automated Verification Tests:**
   ```bash
   pytest tests/test_pipeline.py
   ```
   *Expected output: 6 passed.*
