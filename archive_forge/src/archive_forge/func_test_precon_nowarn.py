import pytest
from ase.atoms import Atoms
from ase.optimize.precon import PreconLBFGS
def test_precon_nowarn():
    PreconLBFGS(Atoms('100H'))