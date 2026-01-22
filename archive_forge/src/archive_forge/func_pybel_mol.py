from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
@property
def pybel_mol(self) -> Molecule:
    """Returns Pybel's Molecule object."""
    return pybel.Molecule(self._ob_mol)