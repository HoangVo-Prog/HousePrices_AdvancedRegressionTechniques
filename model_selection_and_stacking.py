
"""
Model selection + hyperparameter tuning + 3-model stacking for Ames-like house prices.

Now includes 8 models:
- RandomForest, XGB, SVR, ElasticNet, Ridge, Lasso, CatBoost, LightGBM

Usage:
    python model_selection_and_stacking.py
"""

import warnings
warnings.filterwarnings("ignore")

import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, RidgeCV
from sklearn.svm import SVR

# Optional deps
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from house_price_pipeline import make_feature_space

RANDOM_STATE = 42

# ---------------------------
# Helpers
# ---------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def get_scorers():
    # Neg RMSE so higher is better for cross_validate (we'll flip sign later)
    scorers = {
        "neg_rmse": make_scorer(lambda yt, yp: -math.sqrt(mean_squared_error(yt, yp)), greater_is_better=True),
        "r2": "r2",
    }
    return scorers

def build_feature_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Pipeline:
    # Infer column groups from train/test; safe for TE and rare pooling
    return make_feature_space(X_train, X_test)

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
    # if HAS_LGBM:
    #     models["LGBM"] = LGBMRegressor(
    #         n_estimators=5000,
    #         learning_rate=0.03,
    #         num_leaves=31,
    #         max_depth=-1,
    #         min_child_samples=20,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         reg_alpha=0.1,
    #         reg_lambda=1.0,
    #         min_split_gain=0.0,
    #         random_state=RANDOM_STATE,
    #         n_jobs=-1
    #     )
    return models

def evaluate_models(models: dict, X_train, y_train, X_test, y_test, feature_pipe: Pipeline, cv_splits=5, out_prefix="baseline"):
    results = []
    preds_test = {}

    print(f"Evaluating {len(models)} models with {cv_splits}-fold CV...")
    for name, est in models.items():
        pipe = Pipeline([("features", feature_pipe), ("model", est)])
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=get_scorers(), return_train_score=False, n_jobs=-1
        )
        mean_neg_rmse = scores["test_neg_rmse"].mean()
        std_neg_rmse = scores["test_neg_rmse"].std()
        mean_r2 = scores["test_r2"].mean()
        std_r2 = scores["test_r2"].std()

        # Fit once on full train and evaluate on test
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        test_rmse = rmse(y_test, yp)
        test_r2 = r2_score(y_test, yp)

        results.append({
            "model": name,
            "cv_rmse_mean": -mean_neg_rmse,
            "cv_rmse_std": std_neg_rmse,
            "cv_r2_mean": mean_r2,
            "cv_r2_std": std_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })
        preds_test[name] = yp

        print(f"{name:<14s}  CV RMSE={-mean_neg_rmse:.4f}  CV R2={mean_r2:.4f}  | Test RMSE={test_rmse:.4f}  Test R2={test_r2:.4f}")

    res_df = pd.DataFrame(results).sort_values("cv_rmse_mean")
    # Save
    res_df.to_csv(f"{out_prefix}_model_cv_test_results.csv", index=False)

    # Plots
    plt.figure(figsize=(10,5))
    idx = np.arange(len(res_df))
    plt.bar(idx, res_df["cv_rmse_mean"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("CV RMSE (lower is better)")
    plt.title("Baseline cross-validated RMSE by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cv_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.bar(idx, res_df["cv_r2_mean"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("CV R² (higher is better)")
    plt.title("Baseline cross-validated R² by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cv_r2.png", dpi=150)
    plt.close()

    # Test plots
    plt.figure(figsize=(10,5))
    plt.bar(idx, res_df["test_rmse"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_test_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.bar(idx, res_df["test_r2"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("Test R²")
    plt.title("Test R² by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_test_r2.png", dpi=150)
    plt.close()

    # Save predictions
    preds_df = pd.DataFrame(preds_test)
    preds_df.to_csv(f"{out_prefix}_test_predictions.csv", index=False)

    return res_df, preds_df

# ---------------------------
# Optuna tuning
# ---------------------------
def make_objective(name, X_train, y_train, feature_pipe):
    def objective(trial):
        if name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=100)
            max_depth = trial.suggest_int("max_depth", 4, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        elif name == "ElasticNet":
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000, random_state=RANDOM_STATE)

        elif name == "Ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

        elif name == "Lasso":
            alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
            model = Lasso(alpha=alpha, max_iter=30000, random_state=RANDOM_STATE)

        elif name == "SVR":
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)

        elif name == "XGB" and HAS_XGB:
            learning_rate = trial.suggest_float("learning_rate", 0.005, 0.08, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 7)
            min_child_weight = trial.suggest_float("min_child_weight", 1.0, 8.0)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.95)
            reg_lambda = trial.suggest_float("reg_lambda", 0.3, 10.0, log=True)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            gamma = trial.suggest_float("gamma", 0.0, 0.3)
            max_bin = trial.suggest_int("max_bin", 128, 512)
            model = xgb.XGBRegressor(
                n_estimators=6000,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                gamma=gamma,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                max_bin=max_bin,
                missing=np.nan
            )

        elif name == "CatBoost" and HAS_CAT:
            n_estimators = trial.suggest_int("n_estimators", 1000, 6000, step=500)
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.6, 1.0)
            model = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=n_estimators,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                subsample=subsample,
                colsample_bylevel=colsample_bylevel,
                random_state=RANDOM_STATE,
                verbose=False
            )

        elif name == "LGBM" and HAS_LGBM:
            n_estimators = trial.suggest_int("n_estimators", 2000, 8000, step=500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            num_leaves = trial.suggest_int("num_leaves", 16, 128, step=4)
            max_depth = trial.suggest_int("max_depth", -1, 32)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 0.5)
            model = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_split_gain=min_split_gain,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

        else:
            raise RuntimeError(f"Unknown or unavailable model for tuning: {name}")

        pipe = Pipeline([("features", feature_pipe), ("model", model)])
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=get_scorers(), return_train_score=False, n_jobs=-1
        )
        # Objective: minimize CV RMSE
        cv_rmse = -scores["test_neg_rmse"].mean()
        return cv_rmse
    return objective

def tune_top_models(top_names, X_train, y_train, feature_pipe, n_trials=40):
    tuned = {}
    histories = {}
    for name in top_names:
        if name == "XGB" and not HAS_XGB:
            print("[tune] Skipping XGB (xgboost not installed).")
            continue
        if name == "CatBoost" and not HAS_CAT:
            print("[tune] Skipping CatBoost (catboost not installed).")
            continue
        if name == "LGBM" and not HAS_LGBM:
            print("[tune] Skipping LGBM (lightgbm not installed).")
            continue
        if not HAS_OPTUNA:
            print("[tune] Optuna not installed; skipping tuning for", name)
            continue

        print(f"[tune] Tuning {name} for {n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(name, X_train, y_train, feature_pipe), n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
        histories[name] = [(t.number, t.value) for t in study.trials]

        # Rebuild estimator with best params
        if name == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=best_params.get("n_estimators", 800),
                max_depth=best_params.get("max_depth", None),
                min_samples_split=best_params.get("min_samples_split", 2),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                max_features=best_params.get("max_features", "sqrt"),
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        elif name == "ElasticNet":
            model = ElasticNet(
                alpha=best_params.get("alpha", 0.01),
                l1_ratio=best_params.get("l1_ratio", 0.5),
                max_iter=1000000,
                random_state=RANDOM_STATE
            )
        elif name == "Ridge":
            model = Ridge(alpha=best_params.get("alpha", 10.0), random_state=RANDOM_STATE)
        elif name == "Lasso":
            model = Lasso(alpha=best_params.get("alpha", 0.0005), max_iter=1000000, random_state=RANDOM_STATE)
        elif name == "SVR":
            model = SVR(
                kernel="rbf",
                C=best_params.get("C", 10.0),
                epsilon=best_params.get("epsilon", 0.1),
                gamma=best_params.get("gamma", "scale"),
            )
        elif name == "XGB" and HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=6000,
                learning_rate=best_params.get("learning_rate", 0.03),
                max_depth=best_params.get("max_depth", 4),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 2.0),
                reg_lambda=best_params.get("reg_lambda", 3.0),
                reg_alpha=best_params.get("reg_alpha", 0.2),
                gamma=best_params.get("gamma", 0.05),
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                max_bin=best_params.get("max_bin", 256),
                missing=np.nan
            )
        elif name == "CatBoost" and HAS_CAT:
            model = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=best_params.get("n_estimators", 3000),
                depth=best_params.get("depth", 6),
                learning_rate=best_params.get("learning_rate", 0.05),
                l2_leaf_reg=best_params.get("l2_leaf_reg", 3.0),
                subsample=best_params.get("subsample", 0.8),
                colsample_bylevel=best_params.get("colsample_bylevel", 0.8),
                random_state=RANDOM_STATE,
                verbose=False
            )
        elif name == "LGBM" and HAS_LGBM:
            model = LGBMRegressor(
                n_estimators=best_params.get("n_estimators", 5000),
                learning_rate=best_params.get("learning_rate", 0.03),
                num_leaves=best_params.get("num_leaves", 31),
                max_depth=best_params.get("max_depth", -1),
                min_child_samples=best_params.get("min_child_samples", 20),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                reg_alpha=best_params.get("reg_alpha", 0.1),
                reg_lambda=best_params.get("reg_lambda", 1.0),
                min_split_gain=best_params.get("min_split_gain", 0.0),
                random_state=RANDOM_STATE,
                n_jobs=-1
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

def build_stacking(top3_tuned: dict, feature_pipe: Pipeline):
    estimators = []
    for name, info in top3_tuned.items():
        estimators.append((name, info["estimator"]))
    meta = RidgeCV(alphas=np.logspace(-3, 3, 25))
    stack = Pipeline([
        ("features", feature_pipe),
        ("stack", StackingRegressor(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False))
    ])
    return stack

def main():
    # Load data
    df = pd.read_csv("train-house-prices-advanced-regression-techniques.csv")
    assert "SalePrice" in df.columns, "SalePrice column missing."
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    feature_pipe = build_feature_pipeline(X_train, X_test)

    # 1) Baseline evaluation of all requested models
    models = base_models_dict()
    baseline_df, preds_test = evaluate_models(models, X_train, y_train, X_test, y_test, feature_pipe, out_prefix="baseline")

    # Select top-5 by CV RMSE
    top5 = list(baseline_df.sort_values("cv_rmse_mean")["model"].head(5).values)
    print("Top-5 by CV RMSE:", top5)

    # 2) Tune top-5 with Optuna (if available)
    tuned, histories = tune_top_models(top5, X_train, y_train, feature_pipe, n_trials=40)

    # Save tuning histories
    with open("tuning_histories.json", "w") as f:
        json.dump(histories, f)

    # If no tuning was possible, fall back to baseline top-3
    if not tuned:
        print("No tuned models available; falling back to baseline top-3.")
        top3_names = list(baseline_df.sort_values("cv_rmse_mean")["model"].head(3).values)
        top3_tuned = {name: {"estimator": base_models_dict()[name]} for name in top3_names}
    else:
        # Pick top-3 tuned by best_cv_rmse
        top3_names = sorted(tuned.keys(), key=lambda k: tuned[k]["best_cv_rmse"])[:3]
        top3_tuned = {name: tuned[name] for name in top3_names}

    print("Stacking these 3 models:", top3_names)

    # 3) Build stacking regressor
    stack = build_stacking(top3_tuned, feature_pipe)

    # Fit stack and evaluate
    stack.fit(X_train, y_train)
    ypred_stack = stack.predict(X_test)
    stack_rmse = rmse(y_test, ypred_stack)
    stack_r2 = r2_score(y_test, ypred_stack)
    print(f"[STACK] Test RMSE={stack_rmse:.5f}  Test R2={stack_r2:.5f}")

    # Plot predicted vs true for stack
    plt.figure(figsize=(6,6))
    plt.scatter(y_test.values, ypred_stack, alpha=0.6, edgecolors="none")
    lims = [min(y_test.min(), ypred_stack.min()), max(y_test.max(), ypred_stack.max())]
    plt.plot(lims, lims)
    plt.xlabel("True SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Stacking Regressor: True vs Predicted")
    plt.tight_layout()
    plt.savefig("stack_true_vs_pred.png", dpi=150)
    plt.close()

    # Compare bars: best single vs stack
    best_single = baseline_df.sort_values("test_rmse").iloc[0]
    names = [best_single["model"], "Stack"]
    vals_rmse = [best_single["test_rmse"], stack_rmse]
    vals_r2 = [best_single["test_r2"], stack_r2]

    plt.figure(figsize=(6,4))
    plt.bar(np.arange(len(names)), vals_rmse)
    plt.xticks(np.arange(len(names)), names)
    plt.ylabel("Test RMSE")
    plt.title("Best single vs Stack (Test RMSE)")
    plt.tight_layout()
    plt.savefig("best_vs_stack_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(np.arange(len(names)), vals_r2)
    plt.xticks(np.arange(len(names)), names)
    plt.ylabel("Test R²")
    plt.title("Best single vs Stack (Test R²)")
    plt.tight_layout()
    plt.savefig("best_vs_stack_r2.png", dpi=150)
    plt.close()

    # Save all test predictions (including stack)
    out_preds = preds_test.copy()
    out_preds["Stack"] = ypred_stack
    if "Id" in df.columns:
        out_preds.insert(0, "Id", df.loc[X_test.index, "Id"].values)
    out_preds.to_csv("test_predictions_all_models.csv", index=False)

    # Save summary
    summary = baseline_df.copy()
    stack_row = {
        "model": "STACK(" + "+".join(top3_names) + ")",
        "cv_rmse_mean": np.nan,
        "cv_rmse_std": np.nan,
        "cv_r2_mean": np.nan,
        "cv_r2_std": np.nan,
        "test_rmse": stack_rmse,
        "test_r2": stack_r2,
    }
    summary = pd.concat([summary, pd.DataFrame([stack_row])], ignore_index=True)
    summary.to_csv("final_summary.csv", index=False)

    print("Artifacts written:")
    print(" - baseline_model_cv_test_results.csv")
    print(" - baseline_cv_rmse.png, baseline_cv_r2.png")
    print(" - baseline_test_rmse.png, baseline_test_r2.png")
    print(" - tuning_histories.json (if tuning ran)")
    print(" - stack_true_vs_pred.png")
    print(" - best_vs_stack_rmse.png, best_vs_stack_r2.png")
    print(" - test_predictions_all_models.csv")
    print(" - final_summary.csv")


if __name__ == "__main__":
    main()



