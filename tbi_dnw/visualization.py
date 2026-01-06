import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dcurves import dca
from tbi_dnw.config import PUBLICATION_LABELS

def run_dca_analysis(probabilities_dict, imputation_filter=None,
                     threshold_range=(0, 0.50), title_suffix="", clean_legend=False):
    """
    Run Decision Curve Analysis and generate publication-quality plot.
    """
    if not probabilities_dict:
        print("No probability data available for DCA.")
        return None

    # Filter models
    if imputation_filter:
        filtered_dict = {k: v for k, v in probabilities_dict.items()
                         if imputation_filter in k}
    else:
        filtered_dict = probabilities_dict

    if not filtered_dict:
        print(f"No models found with filter:  {imputation_filter}")
        return None

    # Build master DataFrame
    first_model = list(filtered_dict.keys())[0]
    df_dca = pd.DataFrame({
        'true_label': filtered_dict[first_model]['true_label'].values
    })

    model_cols = []
    for model_name, df_proba in filtered_dict.items():
        # Sanitize column name
        col_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        df_dca[col_name] = df_proba['predicted_proba'].values
        model_cols.append(col_name)

    # Run DCA
    try:
        dca_results = dca(
            data=df_dca,
            outcome='true_label',
            modelnames=model_cols,
            thresholds=np.arange(threshold_range[0], threshold_range[1], 0.01)
        )

        # Create publication-quality plot
        plt.figure(figsize=(10, 7), dpi =300)
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_cols)))

        for idx, model_col in enumerate(model_cols):
            model_data = dca_results[dca_results['model'] == model_col]
            label = model_col.replace('_', ' ')
            if clean_legend:
                label = label.replace('Simple', '').strip()

            plt.plot(
                model_data['threshold'],
                model_data['net_benefit'],
                label=label,
                color=colors[idx],
                linewidth=2
            )

        # Add reference lines
        treat_all = dca_results[dca_results['model'] == 'all']
        treat_none = dca_results[dca_results['model'] == 'none']

        if len(treat_all) > 0:
            plt.plot(
                treat_all['threshold'],
                treat_all['net_benefit'],
                'k--', label='Treat All', linewidth=1.5, alpha=0.7
            )
        if len(treat_none) > 0:
            plt.plot(
                treat_none['threshold'],
                treat_none['net_benefit'],
                'k:', label='Treat None', linewidth=1.5, alpha=0.7
            )

        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title(f'Decision Curve Analysis{title_suffix}', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.xlim([threshold_range[0], 0.4])
        plt.ylim([-0.05, max(0.20, plt.ylim()[1])])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return dca_results

    except Exception as e:
        print(f"DCA Error: {e}")
        return None

def run_comprehensive_xgboost_shap(
    all_fitted_models,
    model_key_substring="XGB",
    X_raw=None,
    save_path="xgb_shap.pkl"
):
    """
    Finds the XGBoost model, computes exact SHAP values, applies labels,
    generates thorough plots, and pickles everything for external portability.
    """
    print("Starting Comprehensive XGBoost SHAP Analysis")

    # Locate Model
    target_key = next((k for k in all_fitted_models.keys() if model_key_substring in k), None)
    if not target_key:
        print(f"Error: No model found matching '{model_key_substring}' in ALL_FITTED_MODELS.")
        return

    print(f"Analyzing Model: {target_key}")
    fitted_object = all_fitted_models[target_key]

    # Extract Pipeline Components
    if hasattr(fitted_object, 'estimator'):
        pipeline = fitted_object.estimator
    elif hasattr(fitted_object, 'base_estimator'):
        pipeline = fitted_object.base_estimator
    else:
        pipeline = fitted_object

    try:
        prep = pipeline.named_steps['prep']
        clf = pipeline.named_steps['clf']
    except KeyError:
        print("Error: Could not find 'prep' or 'clf' steps in pipeline.")
        return

    # Prepare Data & Labels
    print("Transforming data...")
    X_transformed = prep.transform(X_raw)

    try:
        feature_names = list(prep.get_feature_names_out())
    except (AttributeError, ValueError):
        feature_names = list(X_raw.columns)

    clean_feature_names = [PUBLICATION_LABELS.get(f, f) for f in feature_names]

    # Compute Exact SHAP Values
    print("Computing Exact SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_transformed)

    shap_values.feature_names = clean_feature_names

    # Visualization
    print("Generating Global Feature Importance (Beeswarm)...")
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"Feature Importance (Beeswarm): {target_key}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    print("Generating Mean Importance (Bar Chart)...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(f"Mean |SHAP| Value: {target_key}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    # Portable Pickling
    print(f"Pickling data to {save_path}...")

    export_data = {
        'shap_values_object': shap_values,
        'X_transformed': X_transformed,
        'feature_names_raw': feature_names,
        'feature_names_clean': clean_feature_names,
        'model_name': target_key,
        'timestamp': pd.Timestamp.now()
    }

    with open(save_path, 'wb') as f:
        pickle.dump(export_data, f)

    print(f"Analysis complete.")

def run_tabpfn_shap(all_fitted_models, model_key="TabPFN (Simple)", X_raw=None, save_path="shap_results.pkl"):
    """
    Computes SHAP for TabPFN using KernelExplainer (sample) and pickles it.
    """
    if model_key not in all_fitted_models:
        print(f"Skipping TabPFN SHAP: {model_key} not found.")
        return

    print(f"Computing SHAP for {model_key}...")
    model = all_fitted_models[model_key]

    if hasattr(model, 'best_estimator_'):
        pipeline = model.best_estimator_
    elif hasattr(model, 'estimator'):
        pipeline = model.estimator
    else:
        pipeline = model

    if hasattr(pipeline, 'named_steps'):
        prep = pipeline.named_steps.get('prep')
        clf = pipeline.named_steps.get('clf')
    else:
        print("Cannot extract pipeline for TabPFN SHAP")
        return

    X_transformed = prep.transform(X_raw)

    try:
        feature_names = list(prep.get_feature_names_out())
    except (AttributeError, ValueError):
        feature_names = list(X_raw.columns)

    explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_transformed, 100))
    shap_values = explainer.shap_values(X_transformed[:100])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value[1] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_transformed[:100],
        feature_names=feature_names
    )

    with open(save_path, 'wb') as f:
        pickle.dump({
            'shap_values': shap_values,
            'base_values': explanation.base_values,
            'X_data': explanation.data,
            'feature_names': feature_names,
            'model_key': model_key
        }, f)
    print(f"Saved TabPFN SHAP to {save_path}")
