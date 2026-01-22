from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def tersoff_potential(structure):
    """Generate the species, Tersoff potential lines for an oxide structure.

        Args:
            structure: pymatgen Structure
        """
    bv = BVAnalyzer()
    el = [site.specie.symbol for site in structure]
    valences = bv.get_valences(structure)
    el_val_dict = dict(zip(el, valences))
    gin = 'species \n'
    qerf_str = 'qerfc\n'
    for key, value in el_val_dict.items():
        if key != 'O' and value % 1 != 0:
            raise SystemError('Oxide has mixed valence on metal')
        specie_str = f'{key} core {value}\n'
        gin += specie_str
        qerf_str += f'{key} {key} 0.6000 10.0000 \n'
    gin += '# noelectrostatics \n Morse \n'
    met_oxi_ters = TersoffPotential().data
    for key, value in el_val_dict.items():
        if key != 'O':
            metal = f'{key}({int(value)})'
            ters_pot_str = met_oxi_ters[metal]
            gin += ters_pot_str
    gin += qerf_str
    return gin