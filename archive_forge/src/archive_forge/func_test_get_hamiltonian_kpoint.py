import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_hamiltonian_kpoint(wan, rng, std_calculator):
    nwannier = 4
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='random')
    kpts = atoms.cell.bandpath(density=50).cartesian_kpts()
    for kpt_c in kpts:
        H_ww = wanf.get_hamiltonian_kpoint(kpt_c=kpt_c)
        for i in range(nwannier):
            for j in range(i + 1, nwannier):
                assert np.abs(H_ww[i, j]) == pytest.approx(np.abs(H_ww[j, i]))