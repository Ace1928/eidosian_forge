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
def parse_qs_params(self):
    """Parse the DFT parameters (as well as functional, HF, vdW params)."""
    pat = re.compile('\\s+QS\\|\\s+(\\w.*)\\s\\s\\s(.*)$')
    self.read_pattern({'QS': pat}, terminate_on_match=False, postprocess=postprocessor, reverse=False)
    self.data['QS'] = dict(self.data['QS'])
    tmp = {}
    i = 1
    for k in list(self.data['QS']):
        if 'grid_level' in str(k) and 'Number' not in str(k):
            tmp[i] = self.data['QS'].pop(k)
            i += 1
    self.data['QS']['Multi_grid_cutoffs_[a.u.]'] = tmp