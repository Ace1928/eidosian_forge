from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def parse_atomic_kind_info(self):
    """
        Parse info on what atomic kinds are present and what basis/pseudopotential is describing
        each of them.
        """
    kinds = re.compile('Atomic kind: (\\w+)')
    orbital_basis_set = re.compile('Orbital Basis Set\\s+(.+$)')
    potential_information = re.compile('(?:Potential information for\\s+(.+$))|(?:atomic kind are GHOST atoms)')
    auxiliary_basis_set = re.compile('Auxiliary Fit Basis Set\\s+(.+$)')
    core_electrons = re.compile('Total number of core electrons\\s+(\\d+)')
    valence_electrons = re.compile('Total number of valence electrons\\s+(\\d+)')
    pseudo_energy = re.compile('Total Pseudopotential Energy.+(-?\\d+.\\d+)')
    self.read_pattern({'kinds': kinds, 'orbital_basis_set': orbital_basis_set, 'potential_info': potential_information, 'auxiliary_basis_set': auxiliary_basis_set, 'core_electrons': core_electrons, 'valence_electrons': valence_electrons, 'pseudo_energy': pseudo_energy}, terminate_on_match=True, postprocess=str, reverse=False)
    atomic_kind_info = {}
    _kinds = []
    for _ in list(chain.from_iterable(self.data['kinds'])):
        if _ not in _kinds:
            _kinds.append(_)
    for i, kind in enumerate(_kinds):
        atomic_kind_info[kind] = {'orbital_basis_set': self.data.get('orbital_basis_set')[i][0], 'pseudo_potential': self.data.get('potential_info')[i][0], 'kind_number': i + 1}
        try:
            if self.data.get('valence_electrons'):
                tmp = self.data.get('valence_electrons')[i][0]
            elif self.data.get('potential_info')[i][0].upper() == 'NONE':
                tmp = 0
            else:
                tmp = self.data.get('potential_info')[i][0].split('q')[-1]
            atomic_kind_info[kind]['valence_electrons'] = int(tmp)
        except (TypeError, IndexError, ValueError):
            atomic_kind_info[kind]['valence_electrons'] = None
        try:
            atomic_kind_info[kind]['core_electrons'] = int(self.data.get('core_electrons')[i][0])
        except (TypeError, IndexError, ValueError):
            atomic_kind_info[kind]['core_electrons'] = None
        try:
            atomic_kind_info[kind]['auxiliary_basis_set'] = self.data.get('auxiliary_basis_set')[i]
        except (TypeError, IndexError):
            atomic_kind_info[kind]['auxiliary_basis_set'] = None
        try:
            atomic_kind_info[kind]['total_pseudopotential_energy'] = float(self.data.get('total_pseudopotential_energy')[i][0] * Ha_to_eV)
        except (TypeError, IndexError, ValueError):
            atomic_kind_info[kind]['total_pseudopotential_energy'] = None
    with zopen(self.filename, mode='rt') as file:
        j = -1
        lines = file.readlines()
        for k, line in enumerate(lines):
            if 'MOLECULE KIND INFORMATION' in line:
                break
            if 'Atomic kind' in line:
                j += 1
            if 'DFT+U correction' in line:
                atomic_kind_info[_kinds[j]]['DFT_PLUS_U'] = {'L': int(lines[k + 1].split()[-1]), 'U_MINUS_J': float(lines[k + 2].split()[-1])}
            k += 1
    self.data['atomic_kind_info'] = atomic_kind_info