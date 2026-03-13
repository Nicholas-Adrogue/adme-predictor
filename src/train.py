"""
train.py — Train and evaluate ADME prediction models.

Usage:
    python src/train.py --dataset lipophilicity --model rf
    python src/train.py --dataset lipophilicity --model xgb --features combined
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data import get_split
from featurize import featurize_dataset
from evaluate import evaluate_model, plot_predictions, plot_feature_importance


MODELS = {
    "rf": lambda: RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    ),
    "xgb": lambda: XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
}


def train(
    dataset: str = "lipophilicity",
    model_name: str = "rf",
    features: str = "morgan",
    save: bool = True,
):
    """Full training pipeline.

    Args:
        dataset: Dataset name (see data.py).
        model_name: Model type — 'rf' or 'xgb'.
        features: Feature method — 'morgan', 'maccs', 'descriptors', 'combined'.
        save: Whether to save the trained model.

    Returns:
        Tuple of (model, metrics_dict).
    """
    # 1. Load data
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset} with {features} features")
    print(f"{'='*60}\n")

    train_df, valid_df, test_df = get_split(dataset)

    # 2. Featurize
    print("\nFeaturizing molecules...")
    X_train = featurize_dataset(train_df["Drug"].tolist(), method=features)
    X_valid = featurize_dataset(valid_df["Drug"].tolist(), method=features)
    X_test = featurize_dataset(test_df["Drug"].tolist(), method=features)

    y_train = train_df["Y"].values
    y_valid = valid_df["Y"].values
    y_test = test_df["Y"].values

    print(f"Feature dimension: {X_train.shape[1]}")

    # 3. Train
    if model_name == "xgb" and not HAS_XGB:
        print("XGBoost not installed. Falling back to Random Forest.")
        model_name = "rf"

    model = MODELS[model_name]()
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    # 4. Evaluate
    print("\n--- Validation Set ---")
    y_pred_valid = model.predict(X_valid)
    valid_metrics = evaluate_model(y_valid, y_pred_valid)

    print("\n--- Test Set ---")
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred_test)

    # 5. Cross-validation on training set
    print("\n--- 5-Fold Cross-Validation (train) ---")
    cv_scores = cross_val_score(
        MODELS[model_name](), X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1
    )
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 6. Save model & results
    if save:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}_{dataset}_{features}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")

        results = {
            "dataset": dataset,
            "model": model_name,
            "features": features,
            "feature_dim": X_train.shape[1],
            "train_size": len(train_df),
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
        }
        results_path = f"models/{model_name}_{dataset}_{features}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    # 7. Generate plots
    os.makedirs("assets", exist_ok=True)
    plot_predictions(
        y_test, y_pred_test,
        title=f"{model_name.upper()} — {dataset} (test set)",
        save_path=f"assets/{model_name}_{dataset}_{features}_preds.png",
    )

    if model_name == "rf" and features == "descriptors":
        plot_feature_importance(
            model, feature_names=None,
            title=f"Feature Importance — {model_name.upper()}",
            save_path=f"assets/{model_name}_{dataset}_{features}_importance.png",
        )

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ADME predictor")
    parser.add_argument("--dataset", default="lipophilicity", choices=["lipophilicity", "caco2", "solubility"])
    parser.add_argument("--model", default="rf", choices=["rf", "xgb"])
    parser.add_argument("--features", default="morgan", choices=["morgan", "maccs", "descriptors", "combined"])
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        model_name=args.model,
        features=args.features,
        save=not args.no_save,
    )
