from subprocess import check_call, DEVNULL
import pytest
from ase.io.pov import write_pov
from ase.build import molecule
from ase.io.pov import get_bondpairs, set_high_bondorder_pairs
def test_povray_io(testdir, povray_executable):
    H2 = molecule('H2')
    write_pov('H2.pov', H2)
    check_call([povray_executable, 'H2.pov'], stderr=DEVNULL)