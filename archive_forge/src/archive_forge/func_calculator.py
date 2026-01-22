from math import radians, sin, cos
import pytest
from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view
def calculator():
    return factory.calc(task='gradient', theory='scf', charge=-1)