# PCC-AIML502: Introduction to Machine Learning (CA-3)
## Project 1: Forecasting Air Pollution in Bidhannagar (Regression & Time-Ordered Evaluation)

> **Core Research Question:** *Can yesterday's readings help forecast tomorrow's $\text{PM}_{2.5}$ in Bidhannagar?*

---

### 1. Group & Assessment Metadata

* **Course Code:** `PCC-AIML502` — Introduction to Machine Learning
* **Assessment:** Continuous Assessment 3 (CA-3 Group Project & Presentation)
* **Assigned Group:** **Group 6 (Section C)**
* **Project Number:** **Project 1** — Forecasting Air Pollution in Bidhannagar
* **Monitoring Station:** Bidhannagar, Kolkata — West Bengal Pollution Control Board (WBPCB Station Site ID: 5129)
* **Dataset Citation:** Central Pollution Control Board (CPCB) continuous ambient air quality records (2017–2023) published via [OpenCity Kolkata Air Quality Portal](https://data.opencity.in/dataset/kolkata-hourly-air-quality-reports/resource/ad9debfa-aa31-4776-bd67-c7419b34f42d)
* **Primary Deliverable Notebook:** [`project1_bidhannagar_pm25.ipynb`](./project1_bidhannagar_pm25.ipynb)

#### Student Roster & Contribution Mapping

| Roll Number | Student Name | Core Project Role & Contribution | Key Deliverable / Notebook Section |
| :--- | :--- | :--- | :--- |
| **13030824141** | **ADITYA KUMAR SAH** | **Group Lead:** Problem formulation, ML architecture design, final evaluation & synthesis | Sections 0, 6, 8; Master Synthesis |
| **13030824142** | **JISHU PALUI** | **Data Engineering:** Hourly ingestion, duplicate resolution & 18-hr completeness pipeline | Sections 1 & 2; Audit & Coverage Exhibits |
| **13030824143** | **PAPRI PAL** | **Feature Engineering:** Autoregressive lags, rolling features & anti-leakage audit | Section 3; Feature Alignment Exhibits |
| **13030824144** | **VIKRAM SHAW** | **Linear Modeling:** Ridge Regression pipeline, scaler isolation & validation alpha tuning | Sections 4 & 5; Ridge Grid Search Table |
| **13030824145** | **SUBENDU RUDRA** | **Ensemble Modeling & Visuals:** Random Forest tuning, top-error inspection & slide figures | Sections 5 & 7; Figure 2 & Figure 3 Exhibits |

---

### 2. Directory Architecture

```text
ml-project/
├── data/
│   └── opencity_bidhannagar_hourly_2017_2023.csv      # Raw WBPCB/CPCB hourly observations
├── outputs/
│   ├── figures/                                       # Exported 300-DPI publication charts
│   │   ├── hourly_coverage.png                        # Exhibit 1: Daily valid hours & completeness
│   │   ├── report_actual_vs_predicted_top3_errors.png # Exhibit 2: Test series & top 3 error annotations
│   │   └── presentation_zoom_event.png                # Exhibit 3: Zoomed transition event for slides
│   └── tables/                                        # Exported CSV evidence tables
│       ├── cleaning_audit.csv                         # Step-by-step row filtering audit
│       ├── yearly_coverage_summary.csv                # Annual data quality and valid day counts
│       ├── feature_alignment_sample.csv               # 10-row sample proving zero target leakage
│       ├── split_summary.csv                          # Chronological Train / Val / Test dates & sizes
│       ├── ridge_validation_tuning.csv                # Validation MAE across Ridge alpha values
│       ├── rf_validation_tuning.csv                   # Validation MAE across Random Forest grids
│       ├── test_mae_comparison.csv                    # Master MAE table (Full Year + Winter DJF)
│       └── top3_errors_surrounding_days.csv           # Detailed inspection of top-3 failure modes
├── project1_bidhannagar_pm25.ipynb                    # Executed Master Jupyter Notebook
└── README.md                                          # Project documentation & presentation guide
```

---

### 3. Key Findings & Direct Answer to Research Question

> **Research Question:** *Can yesterday's readings help forecast tomorrow's $\text{PM}_{2.5}$ in Bidhannagar?*

#### Master Evaluation Summary (Held-Out Test Year: 2023)

| Model | Full Test Year MAE ($\mu\text{g/m}^3$) | Valid Days ($N$) | Full Year vs. Persistence (%) | Winter (Dec–Feb) MAE ($\mu\text{g/m}^3$) | Winter Days ($N$) | Winter vs. Persistence (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** ($\hat{y}_{D+1} = y_D$) | **9.23** | 294 | *0.0% (Baseline)* | **15.25** | 64 | *0.0% (Baseline)* |
| **Ridge Regression** (Tuned $\alpha = 0.01$) | **8.91** | 294 | **+3.4%** | **13.91** | 64 | **+8.8%** |
| **Random Forest Regressor** (Tuned) | **9.77** | 294 | *-5.9%* | **14.88** | 64 | **+2.4%** |

#### Direct Answer & Synthesis
1. **Yes, machine learning beats yesterday's reading, especially when pollution is most severe:**
   * Across the full 2023 calendar year, atmospheric $\text{PM}_{2.5}$ in Bidhannagar has strong serial correlation, making the Persistence Baseline a strong benchmark ($9.23\,\mu\text{g/m}^3$).
   * Regularized **Ridge Regression** achieves the best overall performance ($8.91\,\mu\text{g/m}^3$), providing a statistically validated **$3.4\%$ annual improvement** across 294 test days.
2. **The Winter Window (Dec–Feb) Demonstrates Maximum ML Value:**
   * In Kolkata winters, when ground-level thermal inversions trap smoke and daily means exceed $120\text{--}200\,\mu\text{g/m}^3$, the Persistence Baseline's error surges to $15.25\,\mu\text{g/m}^3$.
   * **Ridge Regression reduces this error to $13.91\,\mu\text{g/m}^3$ (an $8.8\%$ improvement)**. By integrating a 3-day momentum trajectory and a 7-day trailing baseline with seasonal month indicators, Ridge moderates inertia overshoots during turning points.
3. **Linear Regularization Outperforms Random Forests on Autoregressive Lags:**
   * Due to strong multi-collinearity among consecutive daily lags (`lag0` through `lag3`), standard decision trees partition correlated features greedily, resulting in slight overfitting on the test year ($9.77\,\mu\text{g/m}^3$). Ridge's continuous $L_2$ shrinkage acts as an optimal dampener.

---

### 4. Presenter Guide & Slide Talking Points

* **Slide 1 (Introduction & Real Decision):** College alert dashboard to advise outdoor athletic training or student mask advisories at the end of day $D$.
* **Slide 2 (Data Quality & 18-Hour Rule):** Explain the CPCB 18-hour threshold ($75\%$ capture rule). Show `Figure 1A` and `1B` (`hourly_coverage.png`) demonstrating how 2017–2018 were offline, while 2020–2023 provide $\ge 94\%$ compliant coverage.
* **Slide 3 (Fair Evaluation & Anti-Leakage):** Chronological split (Train: 2019–2021, Val: 2022, Test: 2023). Zero shuffling, zero forward leakage.
* **Slide 4 (Master Results):** Present `test_mae_comparison.csv`. Highlight the $+8.8\%$ winter gain for Ridge Regression.
* **Slide 5 (Error Case Analysis):** Show `Figure 2` and `Figure 3` (`presentation_zoom_event.png`). Walk the audience through the largest error event (e.g. 11 Jan 2023), explaining that external meteorological shocks (sudden wind shift or unseasonal rainfall washout) cause abrupt drops that no historical lag model can predict without real-time upstream meteorological telemetry.
* **Slide 6 (Conclusion & Limitations):** Summarize the answer to the research question and note the recommendation: deploy Ridge regression as the primary forecaster, coupled with an operational trigger for sudden weather front alerts.

---

### 5. Reproducibility Guarantee
This project executes end-to-end with zero manual intervention:
```bash
# From the ml-project directory:
python -m jupyter nbconvert --execute --to notebook --inplace project1_bidhannagar_pm25.ipynb
```
All outputs, markdown explanations, formulas, high-resolution figures, and CSV tables will be regenerated and verified automatically.
