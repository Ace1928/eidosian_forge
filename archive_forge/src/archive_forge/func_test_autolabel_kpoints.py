import numpy as np
import pytest
from ase.lattice import MCLC
def test_autolabel_kpoints(cell):
    kpt0 = np.zeros(3)
    kpt1 = np.ones(3)
    path = cell.bandpath([[kpt0, kpt1]], npoints=17, special_points={})
    assert len(path.kpts == 17)
    assert set(path.special_points) == {'Kpt0', 'Kpt1'}
    assert path.kpts[0] == pytest.approx(kpt0)
    assert path.kpts[-1] == pytest.approx(kpt1)