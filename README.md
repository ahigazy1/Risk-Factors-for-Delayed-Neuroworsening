# Risk Factors for Delayed Neuroworsening after Moderate-to-Severe Traumatic Brain Injury

**Author(s):**
Julian Michael Burwell MS¹, Ahmad Higazy BA¹, Rhudjerry Arnet MS¹, Nia Long BS¹, Sofia Muniz-Nuez BS¹, Sina Hemmer MD²'³, Jeffrey Daniel Oliver MD³'⁴, Clemens Maria Schirmer MD PhD⁵

**Affiliation(s):**
1. Geisinger Commonwealth School of Medicine, Scranton, PA, USA

---

## 1. Overview
This repository contains the source code for the analysis presented in the manuscript **"Risk Factors for Delayed Neuroworsening after Moderate-to-Severe Traumatic Brain Injury."**

The codebase implements a machine learning pipeline to predict delayed neuroworsening (DNW) events. It compares traditional statistical methods (LASSO-penalized Logistic Regression) against modern machine learning algorithms (XGBoost) and transformer-based tabular models (TabPFN). The analysis includes:
* **Nested Cross-Validation** for unbiased performance estimation.
* **SHAP (SHapley Additive exPlanations)** for model interpretability.
* **Decision Curve Analysis (DCA)** for clinical utility assessment.

## 2. Repository Structure
The project is organized into a modular Python package `tbi_dnw` and execution scripts.

```text
├── main.py                     # Primary entry point: runs training, evaluation, and saves results
├── requirements.txt            # Python dependencies
├── tbi_dnw/
│   ├── config.py               # Global configuration (paths, hyperparameters, plot settings)
│   ├── data_loader.py          # Data ingestion and cleaning logic
│   ├── preprocessing.py        # Custom clinical imputers (e.g., GCS imputation)
│   ├── models.py               # Pipeline definitions (LASSO, XGBoost, TabPFN)
│   ├── training.py             # Nested cross-validation and calibration logic
│   ├── evaluation.py           # Metrics (AUC, Brier, F1) and threshold optimization
│   ├── visualization.py        # SHAP and DCA plotting functions
│   └── generate_figures.py     # Script to generate high-resolution manuscript figures
