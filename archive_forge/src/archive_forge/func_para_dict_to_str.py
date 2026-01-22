from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def para_dict_to_str(para, joiner=' '):
    para_str = []
    for par, val in sorted(para.items()):
        if val is None or val == '':
            para_str.append(par)
        elif isinstance(val, dict):
            val_str = para_dict_to_str(val, joiner=',')
            para_str.append(f'{par}=({val_str})')
        else:
            para_str.append(f'{par}={val}')
    return joiner.join(para_str)