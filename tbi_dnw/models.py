from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFpr, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from optuna.distributions import FloatDistribution, IntDistribution
from optuna.samplers import TPESampler, RandomSampler
from optuna.integration import OptunaSearchCV
import optuna

from tbi_dnw.preprocessing import ClinicalGCSImputer

# TabPFN initialization attempt
try:
    from tabpfn import TabPFNClassifier
except Exception as e:
    pass

def get_model_pipeline(
    model_type: str,
    imputation_strategy: str = "Simple",
    hyperparam_scorer: str = "average_precision",
    random_state: int = 17822,
    calibrate: bool = False,
    search_type: str = "Optuna",
    n_iter: int = 50,
    use_univariate: bool = False
):
    base_model_type = model_type.replace("_Uni", "").strip()
    is_tree = base_model_type in ["RF", "XGB", "TabPFN"]
    prep_steps = []

    # Imputation Strategy
    if imputation_strategy == "Simple":
        prep_steps.append(("gcs_impute", ClinicalGCSImputer()))
        prep_steps.append(("imputer", SimpleImputer(strategy="median")))
    elif imputation_strategy == "KNN":
        prep_steps.append(("imputer", KNNImputer(n_neighbors=5)))
    elif imputation_strategy == "Iterative":
        prep_steps.append(("imputer", IterativeImputer(random_state=17822, max_iter=10)))
    else:
        prep_steps.append(("imputer", SimpleImputer(strategy="median")))

    if not is_tree:
        prep_steps.append(("scaler", StandardScaler()))

    preprocessor = Pipeline(prep_steps)

    # Define Classifier & Param Search Space
    clf = None
    param_dist = {}
    prefix = "clf__"

    if base_model_type == "Standard_LR":
        clf = LogisticRegression(penalty=None, max_iter=5000, random_state=random_state)

    elif base_model_type == "LASSO":
        clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=random_state)
        param_dist[f"{prefix}C"] = FloatDistribution(1e-4, 100.0, log=True)

    elif base_model_type == "ElasticNet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            random_state=random_state, n_jobs=1
        )
        param_dist[f"{prefix}C"] = FloatDistribution(1e-4, 100.0, log=True)
        param_dist[f"{prefix}l1_ratio"] = FloatDistribution(0.1, 0.9)

    elif base_model_type == "LASSO_LR":
        lasso_estimator = LogisticRegression(
            penalty="l1", solver = 'liblinear', max_iter=5000, random_state=random_state, class_weight='balanced'
        )
        selector = SelectFromModel(
            lasso_estimator, max_features=None, prefit=False
        )
        clf = LogisticRegression(penalty=None, max_iter=5000, random_state=random_state, class_weight=None)

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("selector", selector),
            ("clf", clf)
        ])

        param_dist["selector__estimator__C"] = FloatDistribution(.001, 1000, log=True)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        sampler = RandomSampler(seed=random_state) if search_type == "Randomized" else TPESampler(seed=random_state)

        return OptunaSearchCV(
            pipeline,
            study=optuna.create_study(direction='maximize', sampler=sampler),
            param_distributions=param_dist,
            n_trials=250,
            cv=inner_cv,
            scoring=hyperparam_scorer,
            random_state=random_state,
            n_jobs=1,
            verbose=0,
        )

    elif base_model_type == "RF":
        clf = RandomForestClassifier(random_state=random_state, n_jobs=1, criterion="log_loss")
        param_dist[f"{prefix}n_estimators"] = IntDistribution(100, 500)
        param_dist[f"{prefix}max_depth"] = IntDistribution(2, 15)
        param_dist[f"{prefix}min_samples_leaf"] = IntDistribution(2, 7)
        param_dist[f"{prefix}max_features"] = FloatDistribution(0.5, 0.8)

    elif base_model_type == "XGB":
        clf = XGBClassifier(
            tree_method="hist", objective='binary:logistic',
            random_state=random_state, n_jobs=1,
        )

        param_dist[f"{prefix}n_estimators"] = IntDistribution(100, 1000)
        param_dist[f"{prefix}max_depth"] = IntDistribution(3, 10)
        param_dist[f"{prefix}learning_rate"] = FloatDistribution(0.0001, 0.3, log=True)
        param_dist[f"{prefix}reg_alpha"] = FloatDistribution(0.001, 10.0, log=True)
        param_dist[f"{prefix}reg_lambda"] = FloatDistribution(0.001, 10.0, log=True)
        param_dist[f"{prefix}subsample"] = FloatDistribution(0.1, 1.0)
        param_dist[f"{prefix}colsample_bytree"] = FloatDistribution(0.1, 1.0)
        param_dist[f"{prefix}min_child_weight"] = IntDistribution(3, 10)

    elif base_model_type == "TabPFN":
        clf = TabPFNClassifier(
             random_state=random_state, n_jobs=20, eval_metric='log_loss',
        )
        return Pipeline([("prep", preprocessor), ("clf", clf)])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Construct Pipeline (Standard Case)
    steps = [("prep", preprocessor)]

    if use_univariate:
        selector = SelectFpr(score_func=f_classif, alpha=0.20)
        steps.append(("select", selector))

    steps.append(("clf", clf))
    pipeline = Pipeline(steps)

    # Search Wrapper
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if not param_dist:
        return pipeline

    sampler = RandomSampler(seed=random_state) if search_type == "Randomized" else TPESampler(seed=random_state)

    return OptunaSearchCV(
        pipeline,
        study=optuna.create_study(direction='maximize', sampler=sampler),
        param_distributions=param_dist,
        n_trials=n_iter,
        cv=inner_cv,
        scoring=hyperparam_scorer,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
