# ==============================================================================
# MANUSCRIPT FIGURE GENERATION
# ==============================================================================

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from dcurves import dca
import shap
import warnings
from scipy.stats import spearmanr

from tbi_dnw.config import FIGURE_CONFIG as CONFIG
from tbi_dnw.config import COLORS, SHORT_NAMES, STYLE

warnings.filterwarnings('ignore')

# Style setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': CONFIG['DPI'],
    'savefig.dpi': CONFIG['DPI'],
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
})
sns.set_style("whitegrid")

def save_fig(fig, name):
    """Save figure in configured formats."""
    for fmt in CONFIG['FORMATS']:
        path = os.path.join(CONFIG['OUTPUT_DIR'], f"{name}.{fmt}")
        fig.savefig(path, format=fmt, dpi=CONFIG['DPI'], bbox_inches='tight', facecolor='white')
        print(f"  Saved: {path}")

def get_probas(model_name, all_cv_probabilities):
    """Extract probability array from ALL_CV_PROBABILITIES."""
    val = all_cv_probabilities[model_name]

    # Handle dict with 'probas' key
    if isinstance(val, dict):
        probas = val.get('probas', val.get('y_proba', val.get('predictions', val.get('predicted_proba', None))))
    # Handle DataFrame
    elif hasattr(val, 'values'):
        if 'predicted_proba' in val.columns:
            probas = val['predicted_proba'].values
        else:
            probas = val.values
    else:
        probas = val

    # Convert to numpy if needed
    if hasattr(probas, 'values'):
        probas = probas.values

    # Handle 2D arrays (e.g., shape [n_samples, 2] for binary classification)
    if hasattr(probas, 'ndim') and probas.ndim == 2:
        probas = probas[:, 1]  # Take class 1 probabilities

    # Flatten if needed
    probas = np.ravel(probas)

    return probas

def load_or_compute_shap(model_key, pickle_path, feature_names=None):
    """Load SHAP from pickle (compute logic handled in main.py, so this just loads)."""
    if os.path.exists(pickle_path):
        print(f"Loading SHAP from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            shap_data = pickle.load(f)

        # Handle different pickle formats
        if isinstance(shap_data, shap.Explanation):
            return shap_data
        elif isinstance(shap_data, dict):
            values = None
            for key in ['shap_values', 'shap_values_object', 'values', 'shap', 'explanation']:
                if key in shap_data and shap_data[key] is not None:
                    values = shap_data[key]
                    break

            if values is None:
                print(f"  Warning: Could not find SHAP values in pickle. Keys: {list(shap_data.keys())}")
                return None

            # If feature names are in the pickle, use them
            fnames = shap_data.get('feature_names', shap_data.get('feature_names_clean', feature_names))

            return shap.Explanation(
                values=values,
                base_values=shap_data.get('base_values', shap_data.get('expected_value', 0)),
                data=shap_data.get('X_data', shap_data.get('data', shap_data.get('X_transformed', None))),
                feature_names=fnames
            )
        elif hasattr(shap_data, 'values'):
            return shap_data
        else:
            return shap.Explanation(values=shap_data, feature_names=feature_names)
    else:
        print(f"SHAP pickle not found: {pickle_path}")
        return None

def plot_shap_bar(shap_exp, ax, color, title, top_n=15):
    """Create a clean horizontal bar chart with value labels."""

    # Get mean absolute SHAP values
    if hasattr(shap_exp, 'values'):
        vals = np.abs(shap_exp.values).mean(axis=0)
    else:
        vals = np.abs(shap_exp).mean(axis=0)

    feature_names = shap_exp.feature_names if hasattr(shap_exp, 'feature_names') else [f'Feature {i}' for i in range(len(vals))]

    # Sort and take top N
    sorted_idx = np.argsort(vals)[-top_n:]
    sorted_vals = vals[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # Plot horizontal bars
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_vals, color=color, edgecolor='none', height=0.7)

    # Add value labels on bars
    max_val = sorted_vals.max()
    for i, (bar, val) in enumerate(zip(bars, sorted_vals)):
        if val > max_val * 0.3:  # Inside bar for long bars
            ax.text(val - max_val*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha='right', fontsize=7, color='white', fontweight='bold')
        else:  # Outside bar for short bars
            ax.text(val + max_val*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha='left', fontsize=7, color='black')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=9)
    ax.set_xlim(0, max_val * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=10)

def main():
    print("Starting Figure Generation...")

    # Load CV probabilities
    if not os.path.exists(CONFIG['CV_PROBABILITIES_PICKLE']):
        print(f"Error: {CONFIG['CV_PROBABILITIES_PICKLE']} not found. Run main.py first.")
        return

    with open(CONFIG['CV_PROBABILITIES_PICKLE'], 'rb') as f:
        ALL_CV_PROBABILITIES = pickle.load(f)

    # We need the ground truth 'y'. All models should have the same 'y', so pick one.
    first_model = next(iter(ALL_CV_PROBABILITIES.values()))
    if isinstance(first_model, pd.DataFrame):
        y = first_model['true_label'].values
    else:
        print("Warning: Could not extract ground truth 'y' easily. Check pickle structure.")
        return

    print(f"Loaded CV probabilities for: {list(ALL_CV_PROBABILITIES.keys())}")

    # ==============================================================================
    # FIGURE 4: MODEL PERFORMANCE (ROC, PR, CALIBRATION)
    # ==============================================================================
    print("Generating Figure 4...")
    fig, axes = plt.subplots(1, 3, figsize=(CONFIG['FIGURE_WIDTH']*1.5, 3.5))
    fig.subplots_adjust(wspace=0.35)

    for model in CONFIG['PRIMARY_MODELS']:
        if model not in ALL_CV_PROBABILITIES:
            print(f"Warning: {model} not found, skipping")
            continue

        probas = get_probas(model, ALL_CV_PROBABILITIES)
        color = COLORS.get(model, 'gray')
        name = SHORT_NAMES.get(model, model)
        lw = STYLE['LINE_WIDTH']

        # ROC
        fpr, tpr, _ = roc_curve(y, probas)
        auc = roc_auc_score(y, probas)
        axes[0].plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.2f})', lw=lw)

        # PR
        prec, rec, _ = precision_recall_curve(y, probas)
        ap = average_precision_score(y, probas)
        axes[1].plot(rec, prec, color=color, label=f'{name} (AP={ap:.2f})', lw=lw)

        # Calibration
        prob_true, prob_pred = calibration_curve(y, probas, n_bins=10)
        brier = brier_score_loss(y, probas)
        axes[2].plot(prob_pred, prob_true, 's-', color=color,
                     label=f'{name} (Brier={brier:.3f})',
                     markersize=STYLE['MARKER_SIZE'], lw=lw)

    # Panel labels
    panel_labels = ['A', 'B', 'C']
    for i, (ax, label) in enumerate(zip(axes, panel_labels)):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                fontsize=STYLE['PANEL_LABEL_FONTSIZE'], fontweight='bold', va='bottom')

    # ROC formatting
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves', fontsize=10)
    axes[0].legend(loc='lower right', fontsize=STYLE['LEGEND_FONTSIZE'], framealpha=0.9)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)

    # PR formatting
    prevalence = y.mean()
    axes[1].axhline(prevalence, color='gray', ls='--', alpha=0.4, lw=1, label=f'Baseline ({prevalence:.2f})')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (PPV)')
    axes[1].set_title('Precision-Recall Curves', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=STYLE['LEGEND_FONTSIZE'], framealpha=0.9)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(0, 1.02)

    # Calibration formatting
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1, label='Perfect')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].set_title('Calibration Curves', fontsize=10)
    axes[2].legend(loc='upper left', fontsize=STYLE['LEGEND_FONTSIZE'], framealpha=0.9)
    axes[2].set_xlim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)

    plt.tight_layout()
    save_fig(fig, 'Figure4_Performance')

    # ==============================================================================
    # FIGURE 5: SHAP BAR PLOTS (XGBoost + TabPFN)
    # ==============================================================================
    print("Generating Figure 5...")
    fig, axes = plt.subplots(1, 2, figsize=(CONFIG['FIGURE_WIDTH']*1.4, 6))
    fig.subplots_adjust(wspace=0.5)

    # XGBoost SHAP
    xgb_shap = load_or_compute_shap('XGB (Simple)', CONFIG['XGB_SHAP_PICKLE'])
    if xgb_shap is not None:
        plot_shap_bar(xgb_shap, axes[0], COLORS['XGB (Simple)'], 'XGBoost Feature Importance', CONFIG['SHAP_TOP_FEATURES'])

    # TabPFN SHAP
    tabpfn_shap = load_or_compute_shap('TabPFN (Simple)', CONFIG['TABPFN_SHAP_PICKLE'])
    if tabpfn_shap is not None:
        plot_shap_bar(tabpfn_shap, axes[1], COLORS['TabPFN (Simple)'], 'TabPFN Feature Importance', CONFIG['SHAP_TOP_FEATURES'])

    # Panel labels
    axes[0].text(-0.35, 1.02, 'A', transform=axes[0].transAxes, fontsize=STYLE['PANEL_LABEL_FONTSIZE'], fontweight='bold')
    axes[1].text(-0.35, 1.02, 'B', transform=axes[1].transAxes, fontsize=STYLE['PANEL_LABEL_FONTSIZE'], fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'Figure5_SHAP_Comparison')

    # ==============================================================================
    # Supplemental: Decision Curve Analysis
    # ==============================================================================
    print("Generating Supplemental DCA...")

    # Prepare data - use clean names for legend
    dca_data = {'outcome': y}
    legend_mapping = {}

    for model in CONFIG['PRIMARY_MODELS']:
        if model in ALL_CV_PROBABILITIES:
            display_name = SHORT_NAMES.get(model, model)
            col_name = display_name.replace('-', '_').replace(' ', '_')
            dca_data[col_name] = get_probas(model, ALL_CV_PROBABILITIES)
            legend_mapping[col_name] = display_name

    df_dca = pd.DataFrame(dca_data)
    model_cols = [c for c in df_dca.columns if c != 'outcome']

    # Run DCA
    dca_results = dca(
        data=df_dca,
        outcome='outcome',
        modelnames=model_cols,
        thresholds=np.arange(0, CONFIG['DCA_THRESHOLD_MAX'] + 0.01, 0.01)
    )

    # Plot
    fig, ax = plt.subplots(figsize=(CONFIG['FIGURE_WIDTH'], 4.5))

    col_colors = {
        'LASSO_LR': COLORS['LASSO_LR (Simple)'],
        'XGBoost': COLORS['XGB (Simple)'],
        'TabPFN': COLORS['TabPFN (Simple)'],
    }

    # Plot "Treat All"
    all_data = dca_results[dca_results['model'] == 'all']
    if not all_data.empty:
        ax.plot(all_data['threshold'], all_data['net_benefit'],
                color='#555555', ls='--', lw=1.2, label='Treat All')

    # Plot "Treat None"
    none_data = dca_results[dca_results['model'] == 'none']
    if not none_data.empty:
        ax.plot(none_data['threshold'], none_data['net_benefit'],
                color='#555555', ls=':', lw=1.2, label='Treat None')

    # Plot model curves
    for col in model_cols:
        model_data = dca_results[dca_results['model'] == col]

        # Match back to color key
        color = 'gray'
        for k, c in col_colors.items():
            if k in col:
                color = c
                break

        display_name = legend_mapping.get(col, col)
        ax.plot(model_data['threshold'], model_data['net_benefit'],
                color=color, label=display_name, lw=STYLE['LINE_WIDTH'])

    ax.set_xlabel('Threshold Probability', fontsize=10)
    ax.set_ylabel('Net Benefit', fontsize=10)
    ax.set_title('Decision Curve Analysis', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=STYLE['LEGEND_FONTSIZE'], framealpha=0.9)
    ax.set_xlim(0, CONFIG['DCA_THRESHOLD_MAX'])
    ax.set_ylim(bottom=-0.02)
    ax.axhline(0, color='gray', ls='-', alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'Supplemental_DCA')

    print("Figure generation complete.")

if __name__ == "__main__":
    main()
