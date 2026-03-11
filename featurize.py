"""
featurize.py — Convert SMILES strings to ML-ready features.

Supports multiple molecular representations:
  - Morgan fingerprints (circular fingerprints, similar to ECFP)
  - MACCS keys (predefined structural keys)
  - RDKit descriptors (physicochemical properties)
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem import rdMolDescriptors


def smiles_to_mol(smiles: str):
    """Convert a SMILES string to an RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Generate Morgan (circular) fingerprint as a bit vector.

    Args:
        smiles: SMILES string.
        radius: Radius of the fingerprint (2 ≈ ECFP4, 3 ≈ ECFP6).
        n_bits: Length of the bit vector.

    Returns:
        Numpy array of shape (n_bits,) with 0/1 values.
    """
    mol = smiles_to_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def maccs_keys(smiles: str) -> np.ndarray:
    """Generate MACCS structural keys (166 bits).

    MACCS keys encode the presence/absence of predefined substructural
    patterns. They're less granular than Morgan FPs but more interpretable.
    """
    mol = smiles_to_mol(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def rdkit_descriptors(smiles: str) -> np.ndarray:
    """Compute common RDKit molecular descriptors.

    Returns a vector of physicochemical properties:
      MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds,
      NumAromaticRings, FractionCSP3, NumHeavyAtoms, MolMR
    """
    mol = smiles_to_mol(smiles)

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.MolMR(mol),
    ]
    return np.array(descriptors)


DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
    "NumHeavyAtoms", "MolMR",
]


def featurize_dataset(
    smiles_list: list[str],
    method: str = "morgan",
    **kwargs,
) -> np.ndarray:
    """Featurize a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        method: One of 'morgan', 'maccs', 'descriptors', 'combined'.
        **kwargs: Extra args passed to the featurizer (e.g. radius, n_bits).

    Returns:
        2D numpy array of shape (n_molecules, n_features).
    """
    featurizers = {
        "morgan": lambda s: morgan_fingerprint(s, **kwargs),
        "maccs": maccs_keys,
        "descriptors": rdkit_descriptors,
    }

    if method == "combined":
        # Concatenate Morgan FP + RDKit descriptors
        features = []
        failed = 0
        for smi in smiles_list:
            try:
                morgan = morgan_fingerprint(smi, **kwargs)
                desc = rdkit_descriptors(smi)
                features.append(np.concatenate([morgan, desc]))
            except ValueError:
                features.append(None)
                failed += 1
        if failed > 0:
            print(f"Warning: {failed}/{len(smiles_list)} molecules failed featurization")
        # Replace None with zeros
        dim = len([f for f in features if f is not None][0])
        features = [f if f is not None else np.zeros(dim) for f in features]
        return np.vstack(features)

    if method not in featurizers:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(featurizers.keys()) + ['combined']}")

    func = featurizers[method]
    features = []
    failed = 0

    for smi in smiles_list:
        try:
            features.append(func(smi))
        except ValueError:
            features.append(None)
            failed += 1

    if failed > 0:
        print(f"Warning: {failed}/{len(smiles_list)} molecules failed featurization")

    # Replace failures with zero vectors
    dim = len([f for f in features if f is not None][0])
    features = [f if f is not None else np.zeros(dim) for f in features]

    return np.vstack(features)


if __name__ == "__main__":
    # Quick demo
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"

    print(f"Molecule: Aspirin ({aspirin})")
    print(f"Morgan FP shape:   {morgan_fingerprint(aspirin).shape}")
    print(f"MACCS keys shape:  {maccs_keys(aspirin).shape}")
    print(f"Descriptors shape: {rdkit_descriptors(aspirin).shape}")

    desc = rdkit_descriptors(aspirin)
    for name, val in zip(DESCRIPTOR_NAMES, desc):
        print(f"  {name}: {val:.2f}")
