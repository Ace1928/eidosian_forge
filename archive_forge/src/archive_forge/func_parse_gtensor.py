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
def parse_gtensor(self, gtensor_filename=None):
    """Parse a file containing g tensor."""
    if not gtensor_filename:
        if self.filenames['g_tensor']:
            gtensor_filename = self.filenames['g_tensor'][0]
        else:
            return None
    with zopen(gtensor_filename, mode='rt') as file:
        lines = [line for line in file.read().split('\n') if line]
    data = {}
    data['gmatrix_zke'] = []
    data['gmatrix_so'] = []
    data['gmatrix_soo'] = []
    data['gmatrix_total'] = []
    data['gtensor_total'] = []
    data['delta_g'] = []
    ionic = -1
    dat = None
    for line in lines:
        first = line.strip()
        if first == 'G tensor':
            ionic += 1
            for d in data.values():
                d.append([])
        elif first in data:
            dat = first
        elif first.startswith('delta_g'):
            dat = 'delta_g'
        else:
            splt = [postprocessor(s) for s in line.split()]
            splt = [s for s in splt if isinstance(s, float)]
            data[dat][ionic].append(list(map(float, splt[-3:])))
    self.data.update(data)
    return data['gtensor_total'][-1]