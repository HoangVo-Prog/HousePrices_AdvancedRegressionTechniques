"""
Key fixes for LightGBM in stacking:
1. Reduce n_estimators drastically when used in StackingRegressor (no early stopping available)
2. Add more conservative parameter bounds
3. Implement a custom LightGBM wrapper with internal validation
"""

# Add this new wrapper class before the base_models_dict() function:

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split as tts

class LGBMRegressorWithEarlyStopping(BaseEstimator, RegressorMixin):
    """
    LightGBM wrapper that uses internal train/val split for early stopping.
    This allows early stopping even when used inside StackingRegressor.
    """
    def __init__(self, max_n_estimators=4000, learning_rate=0.03, num_leaves=31,
                 max_depth=12, min_child_samples=20, min_child_weight=1e-3,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                 min_split_gain=0.0, early_stopping_rounds=200, val_size=0.15,
                 random_state=42):
        self.max_n_estimators = max_n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state
        self.model_ = None
        self.best_iteration_ = None
        
    def fit(self, X, y):
        # Guard num_leaves
        num_leaves = self.num_leaves
        if self.max_depth > 0:
            num_leaves = min(num_leaves, 2 ** self.max_depth - 1)
        
        # Internal validation split for early stopping
        X_tr, X_val, y_tr, y_val = tts(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        
        self.model_ = LGBMRegressor(
            n_estimators=self.max_n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=int(num_leaves),
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_split_gain=self.min_split_gain,
            min_data_in_bin=3,
            max_bin=255,
            feature_pre_filter=False,
            force_col_wise=True,
            objective="regression",
            deterministic=True,
            verbosity=-1,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        try:
            self.model_.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        except Exception as e:
            print(f"[LGBM wrapper] Early stopping failed: {e}. Training without validation.")
            # Fallback: train on full data with much fewer iterations
            self.model_.set_params(n_estimators=max(500, self.max_n_estimators // 4))
            self.model_.fit(X, y)
            self.best_iteration_ = None
        
        return self
    
    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet")
        return self.model_.predict(X, num_iteration=self.best_iteration_)
    
    def get_params(self, deep=True):
        return {
            "max_n_estimators": self.max_n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_split_gain": self.min_split_gain,
            "early_stopping_rounds": self.early_stopping_rounds,
            "val_size": self.val_size,
            "random_state": self.random_state
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ---------------------------
# UPDATED: base_models_dict() - Use wrapper for LGBM and reduce n_estimators
# ---------------------------
def base_models_dict():
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=RANDOM_STATE),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.0005, max_iter=20000, random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
    }

    if HAS_XGB:
        models["XGB"] = xgb.XGBRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2.0,
            reg_lambda=3.0,
            reg_alpha=0.2,
            gamma=0.05,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            max_bin=256,
            missing=np.nan
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            loss_function="RMSE",
            n_estimators=3000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_state=RANDOM_STATE,
            verbose=False
        )

    if HAS_LGBM:
        # Use the wrapper instead of raw LGBMRegressor
        models["LGBM"] = LGBMRegressorWithEarlyStopping(
            max_n_estimators=3000,  # Will early stop much earlier typically
            learning_rate=0.03,
            num_leaves=31,
            max_depth=12,
            min_child_samples=20,
            min_child_weight=1e-3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_split_gain=0.0,
            early_stopping_rounds=200,
            val_size=0.15,
            random_state=RANDOM_STATE
        )

    return models


# ---------------------------
# UPDATED: Tuning objective for LGBM - use wrapper
# ---------------------------
def make_objective(name, X_train, y_train, feature_pipe):
    def objective(trial):
        # ... keep all other model cases the same ...
        
        elif name == "LGBM" and HAS_LGBM:
            # Use the wrapper for tuning too
            max_n_estimators = trial.suggest_int("max_n_estimators", 1000, 4000, step=500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            max_depth = trial.suggest_int("max_depth", 6, 14)
            num_leaves = trial.suggest_int("num_leaves", 16, 127)  # More conservative upper bound
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50)
            min_child_weight = trial.suggest_float("min_child_weight", 1e-3, 0.1, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 0.3)
            
            model = LGBMRegressorWithEarlyStopping(
                max_n_estimators=max_n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_split_gain=min_split_gain,
                early_stopping_rounds=150,
                val_size=0.15,
                random_state=RANDOM_STATE
            )
            
            # Standard CV (the wrapper handles early stopping internally)
            pipe = Pipeline([("features", clone(feature_pipe)), ("model", model)])
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv, scoring=get_scorers(),
                return_train_score=False, n_jobs=-1
            )
            cv_rmse = -scores["test_neg_rmse"].mean()
            return cv_rmse
        
        # ... rest of the function ...
    
    return objective


# ---------------------------
# UPDATED: tune_top_models - rebuild with wrapper
# ---------------------------
def tune_top_models(top_names, X_train, y_train, feature_pipe, n_trials=40):
    tuned = {}
    histories = {}
    for name in top_names:
        if not HAS_OPTUNA:
            print("[tune] Optuna not installed; skipping tuning for", name)
            continue
        if name == "XGB" and not HAS_XGB:
            print("[tune] Skipping XGB (xgboost not installed)."); continue
        if name == "CatBoost" and not HAS_CAT:
            print("[tune] Skipping CatBoost (catboost not installed)."); continue
        if name == "LGBM" and not HAS_LGBM:
            print("[tune] Skipping LGBM (lightgbm not installed)."); continue

        print(f"[tune] Tuning {name} for {n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(name, X_train, y_train, feature_pipe), n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
        histories[name] = [(t.number, t.value) for t in study.trials]

        # ... keep all other model rebuilding the same ...
        
        elif name == "LGBM" and HAS_LGBM:
            max_depth = best_params.get("max_depth", 12)
            num_leaves = best_params.get("num_leaves", 31)
            
            model = LGBMRegressorWithEarlyStopping(
                max_n_estimators=best_params.get("max_n_estimators", 3000),
                learning_rate=best_params.get("learning_rate", 0.03),
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=best_params.get("min_child_samples", 20),
                min_child_weight=best_params.get("min_child_weight", 1e-3),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                reg_alpha=best_params.get("reg_alpha", 0.1),
                reg_lambda=best_params.get("reg_lambda", 1.0),
                min_split_gain=best_params.get("min_split_gain", 0.0),
                early_stopping_rounds=150,
                val_size=0.15,
                random_state=RANDOM_STATE
            )
        else:
            continue

        tuned[name] = {
            "estimator": model,
            "best_cv_rmse": best_score,
            "best_params": best_params,
        }
        print(f"[tune] {name}: best CV RMSE={best_score:.5f}, best_params={best_params}")
    return tuned, histories
