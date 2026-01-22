import io
import re
import numpy as np
import pytest
from ase.build import bulk
from ase.io import write
from ase.io.elk import parse_elk_eigval, read_elk
from ase.units import Hartree, Bohr
def test_parse_eigval():
    fd = io.StringIO(mock_elk_eigval_out)
    dct = dict(parse_elk_eigval(fd))
    eig = dct['eigenvalues'] / Hartree
    occ = dct['occupations']
    kpts = dct['ibz_kpoints']
    assert len(eig) == 1
    assert len(occ) == 1
    assert pytest.approx(eig[0]) == [[-1.0, -0.5, 1.0], [1.0, 1.1, 1.2]]
    assert pytest.approx(occ[0]) == [[2.0, 1.5, 0.0], [1.9, 1.8, 1.7]]
    assert pytest.approx(kpts) == [[0.0, 0.0, 0.0], [0.0, 0.1, 0.2]]