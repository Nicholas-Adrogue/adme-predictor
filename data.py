"""
data.py — Load and split ADME datasets from Therapeutics Data Commons.
"""
import pandas as pd
from tdc.single_pred import ADME


DATASETS = {
    "lipophilicity": {"name": "Lipophilicity_AstraZeneca", "task": "regression"},
    "caco2": {"name": "Caco2_Wang", "task": "regression"},
    "solubility": {"name": "Solubility_AqSolDB", "task": "regression"},
}


def load_dataset(dataset_name: str = "lipophilicity") -> pd.DataFrame:
    """Load an ADME dataset from TDC.

    Args:
        dataset_name: One of 'lipophilicity', 'caco2', 'solubility'.

    Returns:
        DataFrame with columns: Drug (SMILES), Drug_ID, Y (target value).
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    data = ADME(name=config["name"])
    df = data.get_data()

    print(f"Loaded {dataset_name}: {len(df)} compounds")
    print(f"  Target (Y) range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")
    print(f"  Target (Y) mean:  {df['Y'].mean():.2f} ± {df['Y'].std():.2f}")

    return df


def get_split(dataset_name: str = "lipophilicity", method: str = "scaffold", seed: int = 42):
    """Get train/valid/test split using TDC's built-in splitting.

    Args:
        dataset_name: Dataset to load.
        method: Split method — 'scaffold' (recommended) or 'random'.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, valid_df, test_df).
    """
    config = DATASETS[dataset_name]
    data = ADME(name=config["name"])
    split = data.get_split(method=method, seed=seed)

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    print(f"Split ({method}): train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    return train_df, valid_df, test_df


if __name__ == "__main__":
    # Quick test
    df = load_dataset("lipophilicity")
    print(f"\nSample SMILES: {df['Drug'].iloc[0]}")
    print(f"Sample target:  {df['Y'].iloc[0]:.3f}")

    train, valid, test = get_split("lipophilicity")
