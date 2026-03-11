"""
predict.py — Run predictions on new molecules.

Usage:
    python src/predict.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --model models/rf_lipophilicity_morgan.pkl
"""
import argparse
import pickle
import numpy as np

from featurize import featurize_dataset, rdkit_descriptors, DESCRIPTOR_NAMES


def load_model(model_path: str):
    """Load a saved model from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(smiles: str | list[str], model, features: str = "morgan") -> np.ndarray:
    """Predict ADME property for one or more SMILES strings.

    Args:
        smiles: A single SMILES string or list of SMILES.
        model: Trained sklearn/xgboost model.
        features: Feature method used during training.

    Returns:
        Array of predicted values.
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    X = featurize_dataset(smiles, method=features)
    predictions = model.predict(X)

    return predictions


def describe_molecule(smiles: str) -> dict:
    """Get a human-readable description of a molecule's properties."""
    desc = rdkit_descriptors(smiles)
    return dict(zip(DESCRIPTOR_NAMES, desc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ADME property for a molecule")
    parser.add_argument("--smiles", required=True, help="SMILES string of the molecule")
    parser.add_argument("--model", default="models/rf_lipophilicity_morgan.pkl", help="Path to saved model")
    parser.add_argument("--features", default="morgan", help="Feature method used during training")
    args = parser.parse_args()

    model = load_model(args.model)
    pred = predict(args.smiles, model, features=args.features)

    print(f"\nMolecule: {args.smiles}")
    print(f"Predicted logD: {pred[0]:.3f}")

    print("\nMolecular properties:")
    props = describe_molecule(args.smiles)
    for name, val in props.items():
        print(f"  {name}: {val:.2f}")
