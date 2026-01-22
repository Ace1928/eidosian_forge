import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
@pytest.mark.parametrize('lat', bravais_lattices())
def test_get_radii(lat, std_calculator, wan):
    if (lat.tocell() == FCC(a=1).tocell()).all() or (lat.tocell() == ORCF(a=1, b=2, c=3).tocell()).all():
        pytest.skip('lattices not supported, yet')
    atoms = molecule('H2', pbc=True)
    atoms.cell = lat.tocell()
    atoms.center(vacuum=3.0)
    calc = std_calculator
    wanf = wan(nwannier=4, fixedstates=2, atoms=atoms, calc=calc, initialwannier='bloch', full_calc=True)
    assert not (wanf.get_radii() == 0).all()