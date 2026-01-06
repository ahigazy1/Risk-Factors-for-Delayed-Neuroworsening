# Configuration for TBI DNW Analysis

GLOBAL_CONFIG = {
    # Data Settings
    'DATA_PATH': 'path/to/your/data.csv',  # Update this to the location of your dataset
    'NA_VALUES': ['Unk', 'unk', 'NA', 'Na', 'na', 'N/A', 'n/a', '', ' ', 'ukn', 'nan', 'Unknown'],
    'TARGET_COL': 'Delayed_NW_YN',

    # Modeling Strategy
    'HYPERPARAM_METRIC': 'neg_log_loss',
    'SEARCH_TYPE': 'Randomized',
    'SEARCH_ITERATIONS': 250,
    'THRESHOLD_OPTIMIZATION_METRIC': 'sens_90_spec',

    'PLOT_SEPARATELY': True,  # False = 4-panel summary per model
    'ENABLE_TABPFN': True,

    # Imputation Settings
    'MAIN_IMPUTATION': 'Simple',
    'RUN_SENSITIVITY': True,
    'SENSITIVITY_STRATS': ['KNN', 'Iterative'],

    # System Settings
    'RANDOM_STATE': 17822,
    'N_JOBS': -1,
    'CV_FOLDS': 5,
    'CALIBRATE_MODELS': False,
    'THRESHOLD_POLICY': 'pooled_oof',
}

# Labels for plotting
PUBLICATION_LABELS = {
    'Legal_Sex': 'Sex (Male)',
    'Age': 'Age (Years)',
    'Age_65_YN': 'Geriatric (Age ≥ 65)',

    # History & Mechanism
    'Hx_AMS_YN': 'Hx: Altered Mental Status',
    'Pretrauma_Antiplalet_Anticoagulant_Use_YN': 'Pre-injury Anticoagulant/Antiplatelet Use',
    'Transfer_YN': 'Inter-facility Transfer',
    'MVA_YN': 'Mechanism: Motor Vehicle Accident',
    'Blunt_Injury_YN': 'Mechanism: Blunt Injury',
    'Polytrauma': 'Polytrauma Presence',

    'Alcohol_YN': 'Alcohol Involvement',
    'BAC_at_admit': 'Blood Alcohol Content (BAC)',

    'LOC_YN': 'Loss of Consciousness',
    'Pupillary_Dysfunction_YN': 'Pupillary Dysfunction',
    'PreHospital_Hypotension_YN': 'Pre-hospital Hypotension',
    'Admission_Systolic': 'Admission SBP (mmHg)',
    'Admission_Diastolic': 'Admission DBP (mmHg)',
    'EMS_GCS': 'GCS Total (EMS)',
    'EMS_MOTOR_GCS': 'GCS Motor (EMS)',
    'Admit_GCS': 'GCS Total (Admission)',
    'Admit_Motor_GCS': 'GCS Motor (Admission)',
    'NSGY_GCS': 'GCS Total (Neurosurgery)',
    'NSGY_Motor_GCS': 'GCS Motor (Neurosurgery)',

    'SDH_YN': 'Subdural Hematoma (SDH)',
    'SAH_YN': 'Subarachnoid Hemorrhage (SAH)',
    'EDH_YN': 'Epidural Hematoma (EDH)',
    'IPH_YN': 'Intraparenchymal Hemorrhage (IPH)',
    'IVH_YN': 'Intraventricular Hemorrhage (IVH)',
    'Multicompartment_Hemorrhage_YN': 'Multi-compartment Hemorrhage',
    'DAI_YN': 'Diffuse Axonal Injury (DAI)',
    'TCE_YN': 'Traumatic Cerebral Edema',
    'Cerebral_Contusion_YN': 'Cerebral Contusion',
    'Negative_YN': 'Negative Neuroimaging Findings',
    'Cranial_Fracture': 'Cranial Fracture',
    'Depressed_Cranial_Fracture_YN': 'Depressed Skull Fracture',
    'Basilar_Cistern_Compression': 'Basal Cistern Compression',
    'Midline_Shift_Greaterthan5mm_YN': 'Midline Shift > 5mm',
    'Average_Rotterdam_Score': 'Rotterdam CT Score'
}

# Figure Generation Configuration
FIGURE_CONFIG = {
    # File paths - SEPARATE PICKLE FILES
    'CV_PROBABILITIES_PICKLE': 'all_cv_probabilities.pkl',
    'FITTED_MODELS_PICKLE': 'all_fitted_models.pkl',
    'MAIN_RESULTS_PICKLE': 'main_results.pkl',
    'TABPFN_SHAP_PICKLE': 'shap_results.pkl',
    'XGB_SHAP_PICKLE': 'xgb_shap.pkl',

    # Output settings
    'OUTPUT_DIR': './',
    'DPI': 400,
    'FORMATS': ['png'],

    # Models to include
    'PRIMARY_MODELS': ['LASSO_LR (Simple)', 'XGB (Simple)', 'TabPFN (Simple)'],

    # SHAP settings
    'SHAP_TOP_FEATURES': 15,
    'COMPUTE_SHAP_IF_MISSING': True,

    # DCA settings
    'DCA_THRESHOLD_MAX': 0.50,

    # Style
    'FIGURE_WIDTH': 7.0,
}

COLORS = {
    'LASSO_LR (Simple)': '#2166AC',   # Deep blue
    'XGB (Simple)': '#B2182B',         # Deep red
    'TabPFN (Simple)': '#1A9850',      # Forest green
}

SHORT_NAMES = {
    'LASSO_LR (Simple)': 'LASSO-LR',
    'XGB (Simple)': 'XGBoost',
    'TabPFN (Simple)': 'TabPFN',
}

STYLE = {
    'LINE_WIDTH': 1.5,
    'MARKER_SIZE': 4,
    'LEGEND_FONTSIZE': 8,
    'PANEL_LABEL_FONTSIZE': 12,
}
