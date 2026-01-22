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
def parse_overlap_condition(self):
    """
        Retrieve the overlap condition number
        """
    overlap_condition = re.compile('\\|A\\|\\*\\|A\\^-1\\|.+=\\s+(-?\\d+\\.\\d+E[+\\-]?\\d+)\\s+Log')
    self.read_pattern({'overlap_condition_number': overlap_condition}, terminate_on_match=True, reverse=False, postprocess=postprocessor)