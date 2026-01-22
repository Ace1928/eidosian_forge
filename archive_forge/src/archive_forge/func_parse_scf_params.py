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
def parse_scf_params(self):
    """
        Retrieve the most import SCF parameters: the max number of scf cycles (max_scf),
        the convergence cutoff for scf (eps_scf),.
        """
    max_scf = re.compile('max_scf:\\s+(\\d+)')
    eps_scf = re.compile('eps_scf:\\s+(\\d+)')
    self.read_pattern({'max_scf': max_scf, 'eps_scf': eps_scf}, terminate_on_match=True, reverse=False)
    self.data['scf'] = {}
    self.data['scf']['max_scf'] = self.data.pop('max_scf')[0][0] if self.data['max_scf'] else None
    self.data['scf']['eps_scf'] = self.data.pop('eps_scf')[0][0] if self.data['eps_scf'] else None