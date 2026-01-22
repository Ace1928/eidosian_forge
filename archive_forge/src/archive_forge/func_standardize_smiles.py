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
def standardize_smiles(smiles):
    """Return a standardized canonical SMILES string given a SMILES string.

    Note: This is a convenience function for quickly standardizing a single SMILES string. It is more efficient to use
    the :class:`~molvs.standardize.Standardizer` class directly when working with many molecules or when custom options
    are needed.

    :param string smiles: The SMILES for the molecule.
    :returns: The SMILES for the standardized molecule.
    :rtype: string.
    """
    warn(f'The function standardize_smiles is deprecated and will be removed in the next release.', DeprecationWarning, stacklevel=2)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Standardizer().standardize(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)