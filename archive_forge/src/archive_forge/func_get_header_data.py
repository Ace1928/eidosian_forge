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
def get_header_data(atoms):
    return {'pbc': atoms.pbc.copy(), 'numbers': atoms.get_atomic_numbers(), 'masses': atoms.get_masses() if atoms.has('masses') else None, 'constraints': list(atoms.constraints)}