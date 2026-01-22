from random import randint
from typing import Dict, Tuple, Any
import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.calculators.calculator import (get_calculator_class, all_properties,
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols, atomic_masses
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import decode
@property
def charge(self):
    """Total charge."""
    charges = self.get('inital_charges')
    if charges is None:
        return 0.0
    return charges.sum()