from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
@property
def pymatgen_mol(self) -> Molecule:
    """Returns pymatgen Molecule object."""
    sp = []
    coords = []
    for atom in openbabel.OBMolAtomIter(self._ob_mol):
        sp.append(atom.GetAtomicNum())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    return Molecule(sp, coords)