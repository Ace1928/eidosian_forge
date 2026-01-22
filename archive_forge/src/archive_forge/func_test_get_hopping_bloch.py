import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_hopping_bloch(wan):
    nwannier = 4
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    hop0_ww = wanf.get_hopping([0, 0, 0])
    hop1_ww = wanf.get_hopping([1, 1, 1])
    for i in range(nwannier):
        assert hop0_ww[i, i] != 0
        assert hop1_ww[i, i] != 0
        assert np.abs(hop1_ww[i, i]) < np.abs(hop0_ww[i, i])
        for j in range(i + 1, nwannier):
            assert hop0_ww[i, j] == 0
            assert hop1_ww[i, j] == 0
            assert hop0_ww[i, j] == hop0_ww[j, i]
            assert hop1_ww[i, j] == hop1_ww[j, i]