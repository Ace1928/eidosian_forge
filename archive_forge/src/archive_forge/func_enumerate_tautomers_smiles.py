from warnings import warn
import copy
import logging
from rdkit import Chem
from .charge import ACID_BASE_PAIRS, CHARGE_CORRECTIONS, Reionizer, Uncharger
from .fragment import PREFER_ORGANIC, FragmentRemover, LargestFragmentChooser
from .metal import MetalDisconnector
from .normalize import MAX_RESTARTS, NORMALIZATIONS, Normalizer
from .tautomer import (MAX_TAUTOMERS, TAUTOMER_SCORES, TAUTOMER_TRANSFORMS, TautomerCanonicalizer,
from .utils import memoized_property
def enumerate_tautomers_smiles(smiles):
    """Return a set of tautomers as SMILES strings, given a SMILES string.

    :param smiles: A SMILES string.
    :returns: A set containing SMILES strings for every possible tautomer.
    :rtype: set of strings.
    """
    warn(f'The function enumerate_tautomers_smiles is deprecated and will be removed in the next release.', DeprecationWarning, stacklevel=2)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Standardizer().standardize(mol)
    tautomers = TautomerEnumerator().enumerate(mol)
    return {Chem.MolToSmiles(m, isomericSmiles=True) for m in tautomers}