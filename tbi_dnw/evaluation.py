import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, fbeta_score, f1_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve, roc_auc_score, brier_score_loss
)
from scipy.stats import bootstrap
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tbi_dnw.config import PUBLICATION_LABELS

def compute_metrics_at_threshold(y_true, probas, threshold: float) -> dict:
    """
    Compute classification metrics at a given probability threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    probas = np.asarray(probas, dtype=float)

    # Hard classification at the provided threshold
    y_pred = (probas >= float(threshold)).astype(int)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Clinical Metric Calculations (Safe Division)
    recall = tp / (tp + fn) if (tp + fn) else 0.0            # Sensitivity
    precision = tp / (tp + fp) if (tp + fp) else 0.0         # PPV
    specificity = tn / (tn + fp) if (tn + fp) else 0.0       # Specificity
    npv = tn / (tn + fn) if (tn + fn) else 0.0               # NPV

    # Scoring metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    youdens_j = recall + specificity - 1.0

    return {
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1),
        "f2": float(f2),
        "balanced_accuracy": float(bal_acc),
        "youdens_j": float(youdens_j),
    }

def find_best_threshold(y_true, probas, metric_name='sens_90_spec'):
    """
    Vectorized threshold optimization using sklearn built-ins.
    """
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)

    # Strategy 1: Constrained Optimization (Sens >= X)
    if metric_name in ['sens_90_spec', 'sens_95_spec']:
        fpr, tpr, thresholds = roc_curve(y_true, probas)

        target_sens = 0.95 if metric_name == 'sens_95_spec' else 0.90
        valid_indices = np.where(tpr >= target_sens)[0]

        if len(valid_indices) > 0:
            best_idx = valid_indices[0]
            best_thresh = thresholds[best_idx]
            best_score = 1 - fpr[best_idx]
            return best_thresh, best_score
        else:
            print(f"Warning: No threshold met Sensitivity >= {target_sens}. Falling back to max F2.")
            return find_best_threshold(y_true, probas, metric_name='f2')

    # Strategy 2: Maximization (F1, F2, etc.)
    prec, rec, thresholds = precision_recall_curve(y_true, probas)
    fpr, tpr, roc_thresholds = roc_curve(y_true, probas)
    prec = prec[:-1]
    rec = rec[:-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        if metric_name == 'f1':
            scores = 2 * (prec * rec) / (prec + rec)
        elif metric_name == 'f2':
            beta = 2
            scores = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
        elif metric_name == 'youdens_j':
            scores = tpr - fpr
            best_idx = np.argmax(scores)
            return roc_thresholds[best_idx], scores[best_idx]
        else:
            return 0.5, 0.0

    scores = np.nan_to_num(scores)
    best_idx = np.argmax(scores)

    return thresholds[best_idx], scores[best_idx]

def compute_bootstrap_ci(y_true, y_pred_or_proba, metric_func, n_bootstraps=1000, rng_seed=17822):
    """Compute 95% bootstrap CI using scipy."""
    y_true = np.asarray(y_true)
    y_pred_or_proba = np.asarray(y_pred_or_proba)

    data = (y_true, y_pred_or_proba)

    def stat_func(y_t, y_p, axis=None):
        try:
            return metric_func(y_t.ravel(), y_p.ravel())
        except Exception:
            return np.nan

    try:
        result = bootstrap(
            data,
            statistic=stat_func,
            n_resamples=n_bootstraps,
            random_state=rng_seed,
            paired=True,
            method='percentile',
            vectorized=False
        )
        return result.confidence_interval.low, result.confidence_interval.high
    except Exception:
        return (np.nan, np.nan)

def format_ci(score, low, high):
    """Helper to format score [CI] string."""
    return f"{score:.3f} [{low:.3f} - {high:.3f}]"

def assess_calibration_paper_style(y_true, probs, display_name):
    """
    Computes calibration metrics described in the paper.
    """
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    log_odds = np.log(probs / (1 - probs))

    # Calibration-in-the-Large (CITL)
    exog_citl = np.ones((len(y_true), 1))
    model_citl = sm.GLM(y_true, exog_citl, family=Binomial(link=logit()), offset=log_odds)
    res_citl = model_citl.fit()
    citl_val = res_citl.params[0]
    citl_ci = res_citl.conf_int().iloc[0]

    # Weak Calibration
    exog_weak = sm.add_constant(log_odds)
    model_weak = sm.GLM(y_true, exog_weak, family=Binomial(link=logit()))
    res_weak = model_weak.fit()

    weak_intercept = res_weak.params[0]
    weak_intercept_ci = res_weak.conf_int().iloc[0]

    weak_slope = res_weak.params[1]
    weak_slope_ci = res_weak.conf_int().iloc[1]

    print(f"\n--- {display_name} ---")
    print(f"1. Calibration-in-the-large (Target: Intercept=0)")
    print(f"   Intercept = {citl_val:.2f} (95% CI {citl_ci[0]:.2f} - {citl_ci[1]:.2f})")

    print(f"2. Weak Calibration (Target: Intercept=0, Slope=1)")
    print(f"   Intercept = {weak_intercept:.2f} (95% CI {weak_intercept_ci[0]:.2f} - {weak_intercept_ci[1]:.2f})")
    print(f"   Slope     = {weak_slope:.2f} (95% CI {weak_slope_ci[0]:.2f} - {weak_slope_ci[1]:.2f})")

def get_publishable_odds_ratios(fitted_model, X_raw, y_raw, confidence_level=0.95):
    """
    Extracts features selected by LASSO, calculates Univariable statistics,
    refits a Multivariable logistic regression, and returns a combined report.
    """
    pipeline = fitted_model
    if hasattr(fitted_model, 'best_estimator_'):
        pipeline = fitted_model.best_estimator_

    if not hasattr(pipeline, 'named_steps'):
        print(f"Warning: Model object {type(pipeline).__name__} does not have 'named_steps'. Skipping.")
        return None, None

    if 'selector' not in pipeline.named_steps:
        print("Warning: Pipeline does not have a 'selector' step. Skipping.")
        return None, None

    selector = pipeline.named_steps['selector']
    preprocessor = pipeline.named_steps['prep']

    try:
        all_feat_names = preprocessor.get_feature_names_out()
    except:
        if hasattr(X_raw, 'columns'):
            all_feat_names = X_raw.columns
        else:
            all_feat_names = [f"Var_{i}" for i in range(X_raw.shape[1])]

    support_mask = selector.get_support()
    selected_features = [name for name, is_kept in zip(all_feat_names, support_mask) if is_kept]

    print(f"LASSO Selection: Kept {len(selected_features)} features.")

    # Data Preparation (Unscaled)
    if hasattr(preprocessor, 'named_steps') and 'imputer' in preprocessor.named_steps:
        imputer = preprocessor.named_steps['imputer']
    else:
        imputer = preprocessor.steps[0][1]

    X_imputed_array = imputer.transform(X_raw)
    X_imputed_df = pd.DataFrame(X_imputed_array, columns=all_feat_names)

    X_final_multi = X_imputed_df[selected_features].copy()
    X_final_multi = sm.add_constant(X_final_multi)

    # Univariable Analysis
    print("Calculating Univariable Statistics...")
    uni_results = []

    for feature in selected_features:
        try:
            X_uni = X_imputed_df[[feature]].copy()
            X_uni = sm.add_constant(X_uni)

            model_uni = sm.Logit(y_raw, X_uni)
            result_uni = model_uni.fit(disp=0, method='bfgs')

            params = result_uni.params[feature]
            conf = result_uni.conf_int(alpha=1-confidence_level).loc[feature]
            pval = result_uni.pvalues[feature]

            uni_results.append({
                'Feature': feature,
                'Uni_OR': np.exp(params),
                'Uni_OR_Lower': np.exp(conf[0]),
                'Uni_OR_Upper': np.exp(conf[1]),
                'Uni_P': pval
            })
        except Exception as e:
            print(f"Warning: Univariable fit failed for {feature}: {e}")
            uni_results.append({
                'Feature': feature, 'Uni_OR': np.nan, 'Uni_OR_Lower': np.nan,
                'Uni_OR_Upper': np.nan, 'Uni_P': np.nan
            })

    uni_df = pd.DataFrame(uni_results)

    # Multivariable Analysis
    print("Fitting Multivariable Logistic Regression...")
    try:
        model_multi = sm.Logit(y_raw, X_final_multi)
        result_multi = model_multi.fit(method='bfgs', maxiter=1000, disp=0)
    except Exception as e:
        print(f"Convergence warning: {e}. Retrying with lbfgs...")
        result_multi = model_multi.fit(method='lbfgs', maxiter=2000, disp=0)

    params = result_multi.params
    conf = result_multi.conf_int(alpha=1-confidence_level)
    pvalues = result_multi.pvalues
    bse = result_multi.bse

    multi_df = pd.DataFrame({
        'Feature': params.index,
        'Multi_Coeff': params.values,
        'Multi_SE': bse.values,
        'Multi_P': pvalues.values,
        'Multi_OR': np.exp(params.values),
        'Multi_OR_Lower': np.exp(conf[0].values),
        'Multi_OR_Upper': np.exp(conf[1].values)
    })

    multi_df = multi_df[multi_df['Feature'] != 'const']

    # Merge and Format
    final_df = pd.merge(multi_df, uni_df, on='Feature', how='left')
    final_df['Label'] = final_df['Feature'].map(PUBLICATION_LABELS).fillna(final_df['Feature'])

    def fmt_or(row, prefix):
        if pd.isna(row[f'{prefix}_OR']): return "-"
        return f"{row[f'{prefix}_OR']:.2f} ({row[f'{prefix}_OR_Lower']:.2f}-{row[f'{prefix}_OR_Upper']:.2f})"

    def fmt_p(p):
        if pd.isna(p): return "-"
        if p < 0.001: return "<0.001"
        return f"{p:.3f}"

    final_df['Univariable OR (95% CI)'] = final_df.apply(lambda x: fmt_or(x, 'Uni'), axis=1)
    final_df['Uni P'] = final_df['Uni_P'].apply(fmt_p)

    final_df['Multivariable OR (95% CI)'] = final_df.apply(lambda x: fmt_or(x, 'Multi'), axis=1)
    final_df['Multi P'] = final_df['Multi_P'].apply(fmt_p)

    vif_data = [variance_inflation_factor(X_final_multi.values, i) for i in range(X_final_multi.shape[1])]
    vif_series = pd.Series(vif_data, index=X_final_multi.columns)
    final_df['VIF'] = final_df['Feature'].map(vif_series)

    preds = result_multi.predict(X_final_multi)
    auc = roc_auc_score(y_raw, preds)
    brier = brier_score_loss(y_raw, preds)

    # Display Output
    print("\n" + "="*90)
    print("MODEL FIT STATISTICS (Multivariable)")
    print("="*90)
    print(f"Observations:        {int(result_multi.nobs)}")
    print(f"Events (Positive):   {int(y_raw.sum())}")
    print(f"Pseudo R-squared:    {result_multi.prsquared:.4f}")
    print(f"AIC:                 {result_multi.aic:.4f}")
    print(f"AUC:                 {auc:.3f}")
    print(f"Brier Score:         {brier:.3f}")

    print("\n" + "="*90)
    print("COMPARATIVE REGRESSION RESULTS")
    print("="*90)

    display_cols = ['Label', 'Univariable OR (95% CI)', 'Uni P', 'Multivariable OR (95% CI)', 'Multi P', 'VIF']
    final_df = final_df.sort_values('Multi_P')

    print(final_df[display_cols].to_string(index=False))

    return final_df, result_multi
