from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
def convert_obatoms_to_molecule(self, atoms: Sequence, residue_name: str | None=None, site_property: str='ff_map') -> Molecule:
    """
        Convert list of openbabel atoms to Molecule.

        Args:
            atoms ([OBAtom]): list of OBAtom objects
            residue_name (str): the key in self.map_residue_to_mol. Used to
                restore the site properties in the final packed molecule.
            site_property (str): the site property to be restored.

        Returns:
            Molecule object
        """
    if residue_name is not None and (not hasattr(self, 'map_residue_to_mol')):
        self._set_residue_map()
    coords = []
    zs = []
    for atm in atoms:
        coords.append(list(atm.coords))
        zs.append(atm.atomicnum)
    mol = Molecule(zs, coords)
    if residue_name is not None:
        props = []
        ref = self.map_residue_to_mol[residue_name].copy()
        assert len(mol) == len(ref)
        assert ref.formula == mol.formula
        for idx, site in enumerate(mol):
            assert site.specie.symbol == ref[idx].specie.symbol
            props.append(getattr(ref[idx], site_property))
        mol.add_site_property(site_property, props)
    return mol