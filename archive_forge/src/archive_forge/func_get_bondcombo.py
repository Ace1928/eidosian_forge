from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def get_bondcombo(atoms, bondcombo_def):
    return sum([defin[2] * atoms.get_distance(*defin[0:2]) for defin in bondcombo_def])