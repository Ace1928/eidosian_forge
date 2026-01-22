from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
def remove_bond(self, idx1: int, idx2: int) -> None:
    """
        Remove a bond from an openbabel molecule.

        Args:
            idx1: The atom index of one of the atoms participating the in bond
            idx2: The atom index of the other atom participating in the bond
        """
    for obbond in openbabel.OBMolBondIter(self._ob_mol):
        if obbond.GetBeginAtomIdx() == idx1 and obbond.GetEndAtomIdx() == idx2 or (obbond.GetBeginAtomIdx() == idx2 and obbond.GetEndAtomIdx() == idx1):
            self._ob_mol.DeleteBond(obbond)