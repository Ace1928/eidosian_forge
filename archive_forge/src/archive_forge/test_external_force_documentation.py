from ase import Atoms
from ase.constraints import ExternalForce, FixBondLength
from ase.optimize import FIRE
from ase.calculators.emt import EMT
from numpy.linalg import norm
Tests for class ExternalForce in ase/constraints.py