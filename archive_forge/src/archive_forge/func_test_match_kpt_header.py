from io import StringIO
import numpy as np
import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.abinit import read_abinit_out, read_eig, match_kpt_header
from ase.units import Hartree, Bohr
def test_match_kpt_header():
    header_line = 'kpt#  12, nband=  5, wtk=  0.02778, kpt=  0.4167  0.4167  0.0833 (reduced coord)\n'
    nbands, weight, vector = match_kpt_header(header_line)
    assert nbands == 5
    assert weight == pytest.approx(0.02778)
    assert vector == pytest.approx([0.4167, 0.4167, 0.0833])