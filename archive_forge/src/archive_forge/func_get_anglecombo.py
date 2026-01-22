from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def get_anglecombo(atoms, anglecombo_def):
    return sum([defin[3] * atoms.get_angle(*defin[0:3]) for defin in anglecombo_def])