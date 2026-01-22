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
def parse_chi_tensor(self, chi_filename=None):
    """Parse the magnetic susceptibility tensor."""
    if not chi_filename:
        if self.filenames['chi_tensor']:
            chi_filename = self.filenames['chi_tensor'][0]
        else:
            return None
    with zopen(chi_filename, mode='rt') as file:
        lines = [line for line in file.read().split('\n') if line]
    data = {}
    data['chi_soft'] = []
    data['chi_local'] = []
    data['chi_total'] = []
    data['chi_total_ppm_cgs'] = []
    data['PV1'] = []
    data['PV2'] = []
    data['PV3'] = []
    data['ISO'] = []
    data['ANISO'] = []
    ionic = -1
    dat = None
    for line in lines:
        first = line.strip()
        if first == 'Magnetic Susceptibility Tensor':
            ionic += 1
            for d in data.values():
                d.append([])
        elif first in data:
            dat = first
        elif 'SOFT' in first:
            dat = 'chi_soft'
        elif 'LOCAL' in first:
            dat = 'chi_local'
        elif 'Total' in first:
            dat = 'chi_total_ppm_cgs' if 'ppm' in first else 'chi_total'
        elif first.startswith('PV1'):
            splt = [postprocessor(s) for s in line.split()]
            splt = [s for s in splt if isinstance(s, float)]
            data['PV1'][ionic] = splt[0]
            data['PV2'][ionic] = splt[1]
            data['PV3'][ionic] = splt[2]
        elif first.startswith('ISO'):
            splt = [postprocessor(s) for s in line.split()]
            splt = [s for s in splt if isinstance(s, float)]
            data['ISO'][ionic] = splt[0]
            data['ANISO'][ionic] = splt[1]
        else:
            splt = [postprocessor(s) for s in line.split()]
            splt = [s for s in splt if isinstance(s, float)]
            data[dat][ionic].append(list(map(float, splt)))
    self.data.update(data)
    return data['chi_total'][-1]