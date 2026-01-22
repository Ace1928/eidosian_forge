from subprocess import check_call, DEVNULL
import pytest
from ase.io.pov import write_pov
from ase.build import molecule
from ase.io.pov import get_bondpairs, set_high_bondorder_pairs
def setbond(target, order):
    high_bondorder_pairs[0, target] = ((0, 0, 0), order, (0.1, -0.2, 0))