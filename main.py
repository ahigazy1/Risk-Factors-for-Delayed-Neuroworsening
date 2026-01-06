import time
import pickle
import optuna
import pandas as pd
from tqdm import tqdm
import sys
import os

from tbi_dnw.config import GLOBAL_CONFIG
from tbi_dnw.data_loader import load_and_clean_data
from tbi_dnw.models import get_model_pipeline
from tbi_dnw.training import evaluate_model_single_pass
from tbi_dnw.evaluation import get_publishable_odds_ratios
from tbi_dnw.visualization import run_dca_analysis, run_comprehensive_xgboost_shap, run_tabpfn_shap

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start_time = time.time()

    # Initialize storage
    all_cv_probabilities = {}
    all_fitted_models = {}
    all_search_objects = {}

    # Load data
    X, y = load_and_clean_data(GLOBAL_CONFIG)
    if X is None:
        print("Data loading failed. Exiting.")
        return

    # Define models
    models_to_run = [
        'LASSO_LR',
        'XGB',
    ]
    if GLOBAL_CONFIG.get('ENABLE_TABPFN', False):
        models_to_run.append('TabPFN')

    print(f"Starting analysis with models: {models_to_run}")

    main_results = {}
    sensitivity_results = {}

    # Run primary analysis
    GLOBAL_CONFIG['CURRENT_IMPUTATION'] = GLOBAL_CONFIG['MAIN_IMPUTATION']

    for m in tqdm(models_to_run, desc="Primary Models"):
        model_display_name = f"{m} ({GLOBAL_CONFIG['MAIN_IMPUTATION']})"
        try:
            res = evaluate_model_single_pass(
                get_model_pipeline,
                model_display_name,
                X, y,
                GLOBAL_CONFIG,
                all_fitted_models,
                all_cv_probabilities,
                all_search_objects
            )
            main_results[model_display_name] = res

        except Exception as e:
            print(f"Error training {model_display_name}: {e}")
            import traceback
            traceback.print_exc()

    # Run sensitivity analysis
    if GLOBAL_CONFIG.get('RUN_SENSITIVITY', False):
        print(f"Starting sensitivity analysis: {GLOBAL_CONFIG['SENSITIVITY_STRATS']}")

        for strat in GLOBAL_CONFIG['SENSITIVITY_STRATS']:
            GLOBAL_CONFIG['CURRENT_IMPUTATION'] = strat
            for m in tqdm(models_to_run, desc=f"Sensitivity: {strat}"):
                model_display_name = f"{m} ({strat})"
                try:
                    res = evaluate_model_single_pass(
                        get_model_pipeline,
                        model_display_name,
                        X, y,
                        GLOBAL_CONFIG,
                        all_fitted_models,
                        all_cv_probabilities,
                        all_search_objects
                    )
                    sensitivity_results[model_display_name] = res
                except Exception as e:
                    print(f"Error training {model_display_name}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    print("Saving results...")
    try:
        with open('all_cv_probabilities.pkl', 'wb') as f: pickle.dump(all_cv_probabilities, f)
        with open('all_fitted_models.pkl', 'wb') as f: pickle.dump(all_fitted_models, f)
        with open('main_results.pkl', 'wb') as f: pickle.dump(main_results, f)
        with open('all_search_objects.pkl', 'wb') as f: pickle.dump(all_search_objects, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving pickle files: {e}")

    # Generate SHAP artifacts
    if 'XGB (Simple)' in all_fitted_models:
        run_comprehensive_xgboost_shap(
            all_fitted_models,
            model_key_substring="XGB",
            X_raw=X,
            save_path="xgb_shap.pkl"
        )

    if 'TabPFN (Simple)' in all_fitted_models:
        run_tabpfn_shap(
            all_fitted_models,
            model_key="TabPFN (Simple)",
            X_raw=X,
            save_path="shap_results.pkl"
        )

    # Generate Table 4
    def format_result_row(name, res, analysis_type):
        row = {
            'Type': analysis_type,
            'Algorithm': name,
            'AUC': f"{res['AUC']:.3f} [{res['AUC_CI_Low']:.2f}-{res['AUC_CI_High']:.2f}]",
            'Average Precision': f"{res['Average Precision']:.3f} [{res['Average Precision_CI_Low']:.2f}-{res['Average Precision_CI_High']:.2f}]",
            'Brier Score': f"{res['Brier Score']:.3f} [{res['Brier Score_CI_Low']:.2f}-{res['Brier Score_CI_High']:.2f}]",
            'F1 (opt)': f"{res['F1 Score']:.3f} [{res['F1 Score_CI_Low']:.2f}-{res['F1 Score_CI_High']:.2f}]",
            'Balanced Accuracy (opt)': f"{res['Balanced Acc']:.3f} [{res['Balanced Acc_CI_Low']:.2f}-{res['Balanced Acc_CI_High']:.2f}]",
            'Recall (opt)': f"{res['Sensitivity']:.3f} [{res['Sensitivity_CI_Low']:.2f}-{res['Sensitivity_CI_High']:.2f}]",
            'Precision (opt)': f"{res['PPV (Precision)']:.3f} [{res['PPV (Precision)_CI_Low']:.2f}-{res['PPV (Precision)_CI_High']:.2f}]",
            'Specificity (opt)': f"{res['Specificity']:.3f} [{res['Specificity_CI_Low']:.2f}-{res['Specificity_CI_High']:.2f}]",
            'Threshold (opt)': f"{res['Threshold (opt)']:.3f}",
            '_sort_auc': res['AUC']
        }
        return row

    all_rows = []
    for name, res in main_results.items():
        all_rows.append(format_result_row(name, res, 'Primary'))
    for name, res in sensitivity_results.items():
        all_rows.append(format_result_row(name, res, 'Sensitivity'))

    df_table4 = pd.DataFrame(all_rows)
    display_cols = [
        'Type', 'Algorithm', 'AUC', 'Average Precision', 'Brier Score',
        'F1 (opt)', 'Balanced Accuracy (opt)',
        'Recall (opt)', 'Precision (opt)', 'Specificity (opt)',
        'Threshold (opt)'
    ]

    if not df_table4.empty:
        df_table4_sorted = df_table4.sort_values(['Type', '_sort_auc'], ascending=[True, False])
        df_table4_sorted = df_table4_sorted.drop(columns=['_sort_auc'])
        df_table4_sorted = df_table4_sorted[display_cols]
        print("\nModel Performance Results:")
        print(df_table4_sorted.to_string())
        df_table4_sorted.to_csv('table4_results.csv', index=False)
        print("Table saved to: table4_results.csv")

    # LASSO Odds Ratios
    lasso_key = next((k for k in all_fitted_models.keys() if 'LASSO_LR' in k), None)
    if lasso_key:
        print(f"Processing model: {lasso_key}")
        if lasso_key in all_search_objects:
             model_to_analyze = all_search_objects[lasso_key]
        else:
             model_to_analyze = all_fitted_models[lasso_key]

        df_or, _ = get_publishable_odds_ratios(model_to_analyze, X, y)
        if df_or is not None:
            df_or.to_csv("Final_Publication_Odds_Ratios.csv", index=False)
            print("Results saved to 'Final_Publication_Odds_Ratios.csv'")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    main()
