from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
def delta_func(x, ismear):
    """Replication of VASP's delta function."""
    if ismear < -1:
        raise ValueError('Delta function not implemented for ismear < -1')
    if ismear == -1:
        return step_func(x, -1) * (1 - step_func(x, -1))
    if ismear == 0:
        return np.exp(-(x * x)) / np.sqrt(np.pi)
    return delta_methfessel_paxton(x, ismear)