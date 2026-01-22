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
def constrained_forces(self):
    """Forces after applying constraints."""
    if self._constrained_forces is not None:
        return self._constrained_forces
    forces = self.forces
    constraints = self.constraints
    if constraints:
        forces = forces.copy()
        atoms = self.toatoms()
        for constraint in constraints:
            constraint.adjust_forces(atoms, forces)
    self._constrained_forces = forces
    return forces