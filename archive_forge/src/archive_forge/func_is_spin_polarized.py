import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def is_spin_polarized(self):
    if len(self.chgdiff) > 0:
        return True
    return False