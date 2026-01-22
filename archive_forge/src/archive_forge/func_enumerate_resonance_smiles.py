from warnings import warn
import logging
from rdkit import Chem
def enumerate_resonance_smiles(smiles):
    """Return a set of resonance forms as SMILES strings, given a SMILES string.

    :param smiles: A SMILES string.
    :returns: A set containing SMILES strings for every possible resonance form.
    :rtype: set of strings.
    """
    mol = Chem.MolFromSmiles(smiles)
    mesomers = ResonanceEnumerator().enumerate(mol)
    return {Chem.MolToSmiles(m, isomericSmiles=True) for m in mesomers}