import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
@calc('gpaw')
def test_get_centers(factory):
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(gpts=(32, 32, 32), nbands=4, txt=None)
    atoms = molecule('H2', calculator=calc)
    atoms.center(vacuum=3.0)
    atoms.get_potential_energy()
    wanf = Wannier(nwannier=2, calc=calc, initialwannier='bloch')
    centers = wanf.get_centers()
    com = atoms.get_center_of_mass()
    assert np.abs(centers - [com, com]).max() < 0.0001