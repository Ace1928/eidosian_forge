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
def step_func(x, ismear):
    """Replication of VASP's step function."""
    if ismear < -1:
        raise ValueError('Delta function not implemented for ismear < -1')
    if ismear == -1:
        return 1 / (1.0 + np.exp(-x))
    if ismear == 0:
        return 0.5 + 0.5 * scipy.special.erf(x)
    return step_methfessel_paxton(x, ismear)