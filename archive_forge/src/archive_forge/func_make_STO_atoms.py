import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.build import bulk
from ase.atoms import symbols2numbers
def make_STO_atoms():
    atoms = Atoms(['O', 'O', 'O', 'Sr', 'Ti'], scaled_positions=[[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 0], [0.5, 0.5, 0.5]], cell=[3.905, 3.905, 3.905], pbc=True)
    return atoms