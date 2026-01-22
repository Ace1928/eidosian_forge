import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_hamiltonian_bloch(wan):
    nwannier = 4
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.0)
    kpts = (2, 2, 2)
    Nk = kpts[0] * kpts[1] * kpts[2]
    wanf = wan(atoms=atoms, kpts=kpts, nwannier=nwannier, initialwannier='bloch')
    for k in range(Nk):
        H_ww = wanf.get_hamiltonian(k=k)
        for i in range(nwannier):
            assert H_ww[i, i] != 0
            for j in range(i + 1, nwannier):
                assert H_ww[i, j] == 0
                assert H_ww[i, j] == pytest.approx(H_ww[j, i])