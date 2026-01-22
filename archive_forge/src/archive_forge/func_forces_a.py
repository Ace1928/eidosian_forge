import sys
import numpy as np
from math import factorial
from pytest import approx, fixture
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.vibrations.franck_condon import (FranckCondonOverlap,
@fixture(scope='module')
def forces_a(unrelaxed):
    return unrelaxed.get_forces()