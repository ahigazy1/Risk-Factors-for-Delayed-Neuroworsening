import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, precision_score, f1_score, balanced_accuracy_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibrationDisplay

from tbi_dnw.models import get_model_pipeline
from tbi_dnw.evaluation import find_best_threshold, compute_bootstrap_ci, format_ci

def _fit_search_calibrate_predict_one_fold(
    fold_idx,
    train_ix,
    test_ix,
    X,
    y,
    *,
    model_factory_func,
    base_name,
    imp_strategy,
    config,
    search_backend="threading",
    calib_n_jobs=3,
):
    """
    Executes one fold of nested cross-validation.
    """
    # Use per-fold seed for reproducibility
    fold_seed = int(config["RANDOM_STATE"]) + int(fold_idx)

    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train = y[train_ix]

    # Fit & Search
    model_search = model_factory_func(
        base_name,
        imputation_strategy=imp_strategy,
        hyperparam_scorer=config["HYPERPARAM_METRIC"],
        random_state=fold_seed,
        calibrate=False,
        search_type=config.get("SEARCH_TYPE", "Optuna"),
        n_iter=config.get("SEARCH_ITERATIONS", 50)
    )

    with parallel_backend(search_backend):
        with threadpool_limits(limits=1, user_api="blas"):
            model_search.fit(X_train, y_train)

    # Calibrate
    best_pipe = model_search.best_estimator_ if hasattr(model_search, "best_estimator_") else model_search

    if config.get("CALIBRATE_MODELS", False) and "TabPFN" not in base_name:
        calib_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
        final_clf = CalibratedClassifierCV(
            best_pipe,
            method="sigmoid",
            cv=calib_cv,
            n_jobs=int(calib_n_jobs),
        )
        with parallel_backend(search_backend):
            with threadpool_limits(limits=1, user_api="blas"):
                final_clf.fit(X_train, y_train)
    else:
        final_clf = best_pipe

    # Determine Threshold (On Train Set)
    probas_train = final_clf.predict_proba(X_train)[:, 1]

    train_thresh, _ = find_best_threshold(
        y_train,
        probas_train,
        metric_name=config["THRESHOLD_OPTIMIZATION_METRIC"]
    )

    # Predict (On Test Set)
    probas_test = final_clf.predict_proba(X_test)[:, 1]

    # Apply the training-derived threshold to the test set
    binary_preds_test = (probas_test >= train_thresh).astype(int)

    return test_ix, probas_test, binary_preds_test, train_thresh

def evaluate_model_single_pass(model_factory_func, model_name, X, y, config,
                               all_fitted_models, all_cv_probabilities, all_search_objects):
    """
    Runs 5-fold Nested Cross-Validation.
    Aggregates both probabilities and binary predictions derived from nested thresholding.
    Reports continuous metrics (AUC, AP) and discrete metrics (Sens, Spec) with bootstrap CIs.
    """
    OUTER_FOLDS = config['CV_FOLDS']
    OUTER_N_JOBS = config['N_JOBS']
    SEARCH_BACKEND = "threading"
    CALIB_N_JOBS = 3

    # Parse model name
    if " (" in model_name:
        base_name = model_name.split(" (")[0]
        imp_strategy = model_name.split("(")[1].replace(")", "").strip()
    else:
        base_name = model_name
        imp_strategy = config.get("CURRENT_IMPUTATION", config["MAIN_IMPUTATION"])

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=config["RANDOM_STATE"])

    # Initialize storage arrays
    oof_probas = np.zeros(len(y), dtype=float)
    oof_binary_preds = np.zeros(len(y), dtype=int)
    fold_thresholds = []

    print(f"Training {model_name}...")

    # Cross-Validation Loop
    splits = list(outer_cv.split(X, y))

    if int(OUTER_N_JOBS) == 1:
        # Serial execution
        for fold_idx, (train_ix, test_ix) in enumerate(splits):
            te_ix, proba, bin_pred, thresh = _fit_search_calibrate_predict_one_fold(
                fold_idx, train_ix, test_ix, X, y,
                model_factory_func=model_factory_func,
                base_name=base_name, imp_strategy=imp_strategy,
                config=config, search_backend=SEARCH_BACKEND, calib_n_jobs=CALIB_N_JOBS,
            )
            oof_probas[te_ix] = proba
            oof_binary_preds[te_ix] = bin_pred
            fold_thresholds.append(thresh)
    else:
        # Parallel execution
        with parallel_backend("loky", inner_max_num_threads=1):
            fold_results = Parallel(n_jobs=int(OUTER_N_JOBS), verbose=0)(
                delayed(_fit_search_calibrate_predict_one_fold)(
                    fold_idx, train_ix, test_ix, X, y,
                    model_factory_func=model_factory_func,
                    base_name=base_name, imp_strategy=imp_strategy,
                    config=config, search_backend=SEARCH_BACKEND, calib_n_jobs=CALIB_N_JOBS,
                )
                for fold_idx, (train_ix, test_ix) in enumerate(splits)
            )

        for te_ix, proba, bin_pred, thresh in fold_results:
            oof_probas[te_ix] = proba
            oof_binary_preds[te_ix] = bin_pred
            fold_thresholds.append(thresh)

    threshold_policy = config.get("THRESHOLD_POLICY", "pooled_oof")

    if threshold_policy == "pooled_oof":
        pooled_thresh, _ = find_best_threshold(
            y,
            oof_probas,
            metric_name=config["THRESHOLD_OPTIMIZATION_METRIC"],
        )
        oof_binary_preds = (oof_probas >= pooled_thresh).astype(int)
        fold_thresholds = [float(pooled_thresh)]

    # Final Model Training (Full Data)
    final_search = model_factory_func(
            base_name, imputation_strategy=imp_strategy,
            hyperparam_scorer=config["HYPERPARAM_METRIC"],
            random_state=config["RANDOM_STATE"], calibrate=False,
            search_type=config.get("SEARCH_TYPE", "Optuna"),
            n_iter=config.get("SEARCH_ITERATIONS", 50)
        )

    with parallel_backend(SEARCH_BACKEND):
        with threadpool_limits(limits=1, user_api="blas"):
            final_search.fit(X, y)

    all_search_objects[model_name] = final_search
    best_pipe_full = final_search.best_estimator_ if hasattr(final_search, "best_estimator_") else final_search

    if config.get("CALIBRATE_MODELS", False) and "TabPFN" not in base_name:
        final_model_full = CalibratedClassifierCV(best_pipe_full, method="sigmoid", cv=3, n_jobs=int(CALIB_N_JOBS))
        with parallel_backend(SEARCH_BACKEND):
            with threadpool_limits(limits=1, user_api="blas"):
                final_model_full.fit(X, y)
    else:
        final_model_full = best_pipe_full

    all_fitted_models[model_name] = final_model_full

    # Metrics & Confidence Intervals
    metrics_proba_map = {
        "AUC": roc_auc_score,
        "Average Precision": average_precision_score,
        "Brier Score": brier_score_loss,
    }

    metrics_binary_map = {
        "Sensitivity": recall_score,
        "Specificity": lambda t, p: recall_score(t, p, pos_label=0),
        "PPV (Precision)": precision_score,
        "NPV": lambda t, p: precision_score(t, p, pos_label=0),
        "F1 Score": f1_score,
        "Balanced Acc": balanced_accuracy_score,
    }

    results = {
        "Model": model_name,
        "Threshold (opt)": np.mean(fold_thresholds),
        "Threshold (std)": np.std(fold_thresholds)
    }

    for metric_name, func in metrics_proba_map.items():
        score = func(y, oof_probas)
        ci_low, ci_high = compute_bootstrap_ci(y, oof_probas, func)
        results[metric_name] = score
        results[f"{metric_name}_CI_Low"] = ci_low
        results[f"{metric_name}_CI_High"] = ci_high

    for metric_name, func in metrics_binary_map.items():
        score = func(y, oof_binary_preds)
        ci_low, ci_high = compute_bootstrap_ci(y, oof_binary_preds, func)
        results[metric_name] = score
        results[f"{metric_name}_CI_Low"] = ci_low
        results[f"{metric_name}_CI_High"] = ci_high

    all_cv_probabilities[model_name] = pd.DataFrame({"true_label": y, "predicted_proba": oof_probas})

    # Performance Report
    print(f"\n{'-'*60}")
    print(f"PERFORMANCE REPORT: {model_name}")
    print(f"Validation: Nested Thresholding (Target: {config['THRESHOLD_OPTIMIZATION_METRIC']})")
    print(f"Mean Threshold across folds: {results['Threshold (opt)']:.3f} +/- {results['Threshold (std)']:.3f}")

    print(f"Global Metrics (Probability-based):")
    print(f"  AUC:                 {format_ci(results['AUC'], results['AUC_CI_Low'], results['AUC_CI_High'])}")
    print(f"  Average Precision:   {format_ci(results['Average Precision'], results['Average Precision_CI_Low'], results['Average Precision_CI_High'])}")
    print(f"  Brier Score:         {format_ci(results['Brier Score'], results['Brier Score_CI_Low'], results['Brier Score_CI_High'])}")

    print(f"Clinical Decision Metrics (Nested Binary Predictions):")
    print(f"  Sensitivity:         {format_ci(results['Sensitivity'], results['Sensitivity_CI_Low'], results['Sensitivity_CI_High'])}")
    print(f"  Specificity:         {format_ci(results['Specificity'], results['Specificity_CI_Low'], results['Specificity_CI_High'])}")
    print(f"  PPV (Precision):     {format_ci(results['PPV (Precision)'], results['PPV (Precision)_CI_Low'], results['PPV (Precision)_CI_High'])}")
    print(f"  F1 Score:            {format_ci(results['F1 Score'], results['F1 Score_CI_Low'], results['F1 Score_CI_High'])}")
    print(f"  Balanced Accuracy:   {format_ci(results['Balanced Acc'], results['Balanced Acc_CI_Low'], results['Balanced Acc_CI_High'])}")
    print(f"{'-'*60}\n")

    # Plotting
    plot_name = model_name.replace(" (Simple)", "")

    if config.get("PLOT_SEPARATELY", False):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # ROC
        RocCurveDisplay.from_predictions(y, oof_probas, ax=axes[0, 0], name=plot_name, color="darkorange")
        axes[0, 0].set_title(f"ROC Curve (AUC = {results['AUC']:.3f})")
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)

        # PR
        PrecisionRecallDisplay.from_predictions(y, oof_probas, ax=axes[0, 1], name=plot_name, color="teal")
        axes[0, 1].set_title(f"PR Curve (AP = {results['Average Precision']:.3f})")

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y, oof_binary_preds, ax=axes[1, 0], colorbar=False, cmap="Blues")
        axes[1, 0].set_title(f"Confusion Matrix")

        # Calibration
        CalibrationDisplay.from_predictions(y, oof_probas, ax=axes[1, 1], n_bins=10, name=plot_name)
        axes[1, 1].set_title(f"Calibration (Brier = {results['Brier Score']:.3f})")

        plt.suptitle(f"{plot_name} - Performance Summary", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    return results
