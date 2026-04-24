"""
app.py — Gradio app for ADME property prediction.

Deploy to HuggingFace Spaces:
  1. Create a new Space at huggingface.co/new-space
  2. Choose "Gradio" as the SDK
  3. Upload this file, requirements.txt, and your saved model (.pkl)
"""
import base64
import pickle
from io import BytesIO

import numpy as np
import gradio as gr
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, MACCSkeys

# ── Load your trained model ──────────────────────────────────────────────
MODEL_PATH = "xgb_lipophilicity_combined.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ── Featurization (must match training exactly) ─────────────────────────

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))


def smiles_to_descriptors_array(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
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
    ])


def smiles_to_combined(smiles):
    morgan = smiles_to_morgan(smiles)
    desc = smiles_to_descriptors_array(smiles)
    if morgan is None or desc is None:
        return None
    return np.concatenate([morgan, desc])


def smiles_to_descriptors_display(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    pairs = [
        ("Molecular Weight", f"{Descriptors.MolWt(mol):.1f} g/mol"),
        ("LogP (calculated)", f"{Descriptors.MolLogP(mol):.2f}"),
        ("TPSA", f"{Descriptors.TPSA(mol):.1f} Å²"),
        ("H-Bond Donors", str(Descriptors.NumHDonors(mol))),
        ("H-Bond Acceptors", str(Descriptors.NumHAcceptors(mol))),
        ("Rotatable Bonds", str(Descriptors.NumRotatableBonds(mol))),
        ("Aromatic Rings", str(Descriptors.NumAromaticRings(mol))),
        ("Fraction sp3", f"{Descriptors.FractionCSP3(mol):.3f}"),
        ("Heavy Atoms", str(Descriptors.HeavyAtomCount(mol))),
        ("Molar Refractivity", f"{Descriptors.MolMR(mol):.1f}"),
    ]
    return "\n".join([f"{k}: {v}" for k, v in pairs])


def lipinski_check(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid molecule"
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    violations = []
    if mw > 500:
        violations.append(f"MW = {mw:.0f} (> 500)")
    if logp > 5:
        violations.append(f"LogP = {logp:.2f} (> 5)")
    if hbd > 5:
        violations.append(f"HBD = {hbd} (> 5)")
    if hba > 10:
        violations.append(f"HBA = {hba} (> 10)")

    if len(violations) == 0:
        return "✓ Pass — no violations"
    else:
        return f"✗ {len(violations)} violation(s): " + "; ".join(violations)


def draw_molecule_html(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=(400, 300))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<div style="display:flex;justify-content:center;padding:12px;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="background:white;border-radius:8px;padding:8px;">'
        f'</div>'
    )


# ── Main prediction function ─────────────────────────────────────────────

def predict(smiles):
    smiles = smiles.strip()
    if not smiles:
        return "", "Please enter a SMILES string.", "", ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", "Invalid SMILES string. Please check the input.", "", ""

    features = smiles_to_combined(smiles)
    if features is None:
        return "", "Failed to generate features.", "", ""

    prediction = model.predict(features.reshape(1, -1))[0]

    img_html = draw_molecule_html(smiles)
    descriptors = smiles_to_descriptors_display(smiles)
    lipinski = lipinski_check(smiles)
    pred_text = f"Predicted logD: {prediction:.3f}"

    return img_html, pred_text, descriptors, lipinski


# ── Example molecules ────────────────────────────────────────────────────

MOLECULES = {
    # Common drugs
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Naproxen": "COc1ccc2cc(CC(C)C(=O)O)ccc2c1",
    "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    # Cardiovascular
    "Atorvastatin": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(c2ccc(F)cc2)c(c1c1ccccc1)C(=O)Nc1ccccc1",
    "Warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "Propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    # Hormones
    "Testosterone": "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
    "Estradiol": "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",
    # Antibiotics
    "Ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Trimethoprim": "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
    # Antidepressants
    "Fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "Sertraline": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    # Other
    "Pyrene": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "Nicotine": "CN1CCCC1c1cccnc1",
    "Melatonin": "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
}

MOLECULE_NAMES = list(MOLECULES.keys())


def load_example(name):
    """Load a SMILES string from the example dropdown."""
    if name and name in MOLECULES:
        return MOLECULES[name]
    return ""


# ── Gradio interface ─────────────────────────────────────────────────────

with gr.Blocks(
    title="ADME Property Predictor",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
) as demo:
    gr.Markdown(
        """
        # 🧪 ADME Property Predictor
        Predict lipophilicity (logD) and view molecular descriptors for any drug molecule.

        Enter a SMILES string below or select an example molecule to get started.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            smiles_input = gr.Textbox(
                label="SMILES String",
                placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
                lines=1,
            )
            with gr.Row():
                example_dropdown = gr.Dropdown(
                    choices=MOLECULE_NAMES,
                    label="Example Molecules",
                    value=None,
                    interactive=True,
                )
                predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            mol_image = gr.HTML(label="Molecule Structure")

    example_dropdown.change(
        fn=load_example,
        inputs=example_dropdown,
        outputs=smiles_input,
    )

    with gr.Row():
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", lines=1)
            lipinski_output = gr.Textbox(label="Lipinski Rule of Five", lines=1)
        with gr.Column():
            descriptors_output = gr.Textbox(label="Molecular Descriptors", lines=12)

    predict_btn.click(
        fn=predict,
        inputs=smiles_input,
        outputs=[mol_image, prediction_output, descriptors_output, lipinski_output],
    )
    smiles_input.submit(
        fn=predict,
        inputs=smiles_input,
        outputs=[mol_image, prediction_output, descriptors_output, lipinski_output],
    )

    gr.Markdown(
        """
        ---
        **About:** This model uses XGBoost trained on combined features (Morgan fingerprints + RDKit descriptors)
        from the [Lipophilicity (AstraZeneca)](https://tdcommons.ai/) dataset.
        See the [GitHub repo](https://github.com/YOUR_USERNAME/adme-predictor)
        for the full training pipeline and notebooks.
        """
    )

if __name__ == "__main__":
    demo.launch()
