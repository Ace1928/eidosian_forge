import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so
def test_gamma_pt():
    size, offsets = map(tuple, k2so())
    assert (size, offsets) == ((1, 1, 1), (0, 0, 0))