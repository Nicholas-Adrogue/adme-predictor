"""
evaluate.py — Metrics and visualization for model evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics.

    Returns:
        Dict with RMSE, MAE, R² values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: str | None = None,
):
    """Scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.4, s=20, c="#2563eb", edgecolors="none")

    # Perfect prediction line
    lims = [
        min(y_true.min(), y_pred.min()) - 0.5,
        max(y_true.max(), y_pred.max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect prediction")

    # Metrics annotation
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05, 0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Actual", fontsize=13)
    ax.set_ylabel("Predicted", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.close(fig)
    return fig


def plot_feature_importance(
    model,
    feature_names: list[str] | None = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str | None = None,
):
    """Bar chart of top feature importances (for tree-based models)."""
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.barh(range(len(top_names)), top_values[::-1], color="#2563eb", alpha=0.8)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.close(fig)
    return fig


def compare_models(results: list[dict]) -> None:
    """Print a comparison table of multiple model results."""
    print(f"\n{'Model':<20} {'Features':<15} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 62)
    for r in results:
        m = r["test_metrics"]
        print(f"{r['model']:<20} {r['features']:<15} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")
