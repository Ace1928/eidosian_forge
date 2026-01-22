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
def restore_site_properties(self, site_property: str='ff_map', filename: str | None=None) -> Molecule:
    """
        Restore the site properties for the final packed molecule.

        Args:
            site_property (str):
            filename (str): path to the final packed molecule.

        Returns:
            Molecule
        """
    if not self.control_params['filetype'] == 'pdb':
        raise ValueError('site properties can only be restored for pdb files.')
    filename = filename or self.control_params['output']
    bma = BabelMolAdaptor.from_file(filename, 'pdb')
    pbm = pybel.Molecule(bma._ob_mol)
    assert len(pbm.residues) == sum((x['number'] for x in self.param_list))
    packed_mol = self.convert_obatoms_to_molecule(pbm.residues[0].atoms, residue_name=pbm.residues[0].name, site_property=site_property)
    for resid in pbm.residues[1:]:
        mol = self.convert_obatoms_to_molecule(resid.atoms, residue_name=resid.name, site_property=site_property)
        for site in mol:
            packed_mol.append(site.species, site.coords, properties=site.properties)
    return packed_mol