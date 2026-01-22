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
def super_parent(self, mol, skip_standardize=False):
    """Return the super parent of a given molecule.

        THe super parent is fragment, charge, isotope, stereochemistry and tautomer insensitive. From the input
        molecule, the largest fragment is taken. This is uncharged and then isotope and stereochemistry information is
        discarded. Finally, the canonical tautomer is determined and returned.

        :param mol: The input molecule.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :param bool skip_standardize: Set to True if mol has already been standardized.
        :returns: The super parent molecule.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
    if not skip_standardize:
        mol = self.standardize(mol)
    mol = self.charge_parent(mol, skip_standardize=True)
    mol = self.isotope_parent(mol, skip_standardize=True)
    mol = self.stereo_parent(mol, skip_standardize=True)
    mol = self.tautomer_parent(mol, skip_standardize=True)
    mol = self.standardize(mol)
    return mol