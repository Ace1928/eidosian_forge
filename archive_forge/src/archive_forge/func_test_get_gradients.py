import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_gradients(wan, rng):
    wanf = wan(nwannier=4, fixedstates=2, kpts=(1, 1, 1), initialwannier='bloch', std_calc=False)
    step = rng.rand(wanf.get_gradients().size) + 1j * rng.rand(wanf.get_gradients().size)
    step *= 1e-08
    step -= dagger(step)
    f1 = wanf.get_functional_value()
    wanf.step(step)
    f2 = wanf.get_functional_value()
    assert (np.abs((f2 - f1) / step).ravel() - np.abs(wanf.get_gradients())).max() < 0.0001