# Molecular Descriptors Reference

This project uses the following RDKit molecular descriptors as features for ADME prediction. Each descriptor captures a different aspect of a molecule's physical or chemical properties that influences its pharmacokinetic behavior.

## Descriptors Used

**Molecular Weight (MolWt)** — The total mass of the molecule in g/mol. Heavier molecules tend to have lower oral absorption. Lipinski's Rule of Five suggests drug candidates should stay below 500 g/mol.

**LogP (Calculated)** — The octanol-water partition coefficient, measuring how a molecule distributes between a fatty (octanol) and aqueous (water) phase. Positive values indicate lipophilicity (fat-soluble), negative values indicate hydrophilicity (water-soluble). Calculated here using the Wildman-Crippen method. This measures the neutral form of the molecule only — see logD for the pH-adjusted version.

**TPSA (Topological Polar Surface Area)** — The surface area of the molecule occupied by polar atoms (mainly oxygen and nitrogen) and their attached hydrogens, measured in Å². High TPSA means more polar surface, which generally reduces membrane permeability. Drugs with TPSA under ~140 Å² tend to have better oral absorption.

**H-Bond Donors (NumHDonors)** — The number of NH and OH groups that can donate a hydrogen bond. More donors generally reduce a molecule's ability to cross cell membranes. Lipinski's rule suggests no more than 5 for oral drugs.

**H-Bond Acceptors (NumHAcceptors)** — The number of nitrogen and oxygen atoms that can accept a hydrogen bond. Like donors, too many acceptors can hinder membrane permeability. Lipinski's rule suggests no more than 10.

**Rotatable Bonds (NumRotatableBonds)** — The number of bonds that allow free rotation, excluding bonds in rings and to terminal groups like -CH₃. More rotatable bonds means a more flexible molecule, which can reduce oral bioavailability because the molecule loses more entropy when binding to a target or crossing a membrane.

**Aromatic Rings (NumAromaticRings)** — The count of aromatic ring systems (like benzene rings) in the molecule. Aromatic rings contribute to lipophilicity and can participate in π-stacking interactions with protein binding sites. Too many aromatic rings can reduce solubility and increase off-target binding.

**Fraction sp3 (FractionCSP3)** — The fraction of carbon atoms that are sp3-hybridized (tetrahedral geometry, single bonds only) versus sp2 (flat, as in aromatic rings) or sp (linear). Higher Fsp3 correlates with better solubility and clinical success rates. It's a rough measure of three-dimensionality — flat molecules (low Fsp3) tend to be more promiscuous binders.

**Heavy Atoms (NumHeavyAtoms)** — The total number of non-hydrogen atoms. This is a simple proxy for molecular size, closely correlated with molecular weight but not influenced by the number of hydrogens.

**Molar Refractivity (MolMR)** — A measure of how polarizable a molecule's electron cloud is, calculated from atomic contributions. It correlates with molecular volume and surface area. Higher molar refractivity generally means a larger, more polarizable molecule, which affects how it interacts with biological membranes and protein binding pockets.

## Prediction Target

**LogD** — The distribution coefficient measured at physiological pH (7.4). Unlike logP, which only considers the neutral form of a molecule, logD accounts for the fact that many drugs have ionizable groups (acids and bases) that gain or lose protons at blood pH. This makes logD a more realistic predictor of how a drug partitions between aqueous and lipid environments in the body, directly impacting absorption, distribution, and metabolism.
