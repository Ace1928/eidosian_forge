import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_md_min():
    tol = 1e-08
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=complex), shift=1.0)
    md_min(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-05)