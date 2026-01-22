from typing import Dict, Any
import numpy as np
import ase
from ase.db import connect
from ase.calculators.calculator import Calculator
def atoms_almost_equal(a, b, tol=1e-09):
    return np.abs(a.positions - b.positions).max() < tol and (a.numbers == b.numbers).all() and (np.abs(a.cell - b.cell).max() < tol) and (a.pbc == b.pbc).all()