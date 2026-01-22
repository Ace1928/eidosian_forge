import numpy as np
from ase import Atoms
from ase.calculators.test import FreeElectrons
from ase.dft.kpoints import get_special_points
Test band structure from different variations of hexagonal cells.