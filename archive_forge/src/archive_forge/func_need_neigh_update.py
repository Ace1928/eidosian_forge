import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
@property
def need_neigh_update(self):
    return self.neigh.need_neigh_update