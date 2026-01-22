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
def parse_total_numbers(self):
    """Parse total numbers (not usually important)."""
    atomic_kinds = '- Atomic kinds:\\s+(\\d+)'
    atoms = '- Atoms:\\s+(\\d+)'
    shell_sets = '- Shell sets:\\s+(\\d+)'
    shells = '- Shells:\\s+(\\d+)'
    primitive_funcs = '- Primitive Cartesian functions:\\s+(\\d+)'
    cart_base_funcs = '- Cartesian basis functions:\\s+(\\d+)'
    spher_base_funcs = '- Spherical basis functions:\\s+(\\d+)'
    self.read_pattern({'atomic_kinds': atomic_kinds, 'atoms': atoms, 'shell_sets': shell_sets, 'shells': shells, 'primitive_cartesian_functions': primitive_funcs, 'cartesian_basis_functions': cart_base_funcs, 'spherical_basis_functions': spher_base_funcs}, terminate_on_match=True)