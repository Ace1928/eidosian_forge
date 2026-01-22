import warnings
from typing import Tuple
import numpy as np
from ase import __version__
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.constraints import dict2constraint
from ase.calculators.calculator import PropertyNotImplementedError
from ase.atoms import Atoms
from ase.io.jsonio import encode, decode
from ase.io.pickletrajectory import PickleTrajectory
from ase.parallel import world
from ase.utils import tokenize_version
def headers_equal(headers1, headers2):
    assert len(headers1) == len(headers2)
    eq = True
    for key in headers1:
        eq &= np.array_equal(headers1[key], headers2[key])
    return eq