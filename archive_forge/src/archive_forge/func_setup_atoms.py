from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def setup_atoms():
    atoms = molecule('CH3CH2OH', vacuum=5.0)
    atoms.rattle(stdev=0.3)
    return atoms