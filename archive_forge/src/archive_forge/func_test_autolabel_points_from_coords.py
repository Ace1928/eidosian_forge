import pytest
import numpy as np
from ase.dft.kpoints import resolve_custom_points
@pytest.mark.parametrize('kptcoords', [[np.zeros(3), np.ones(3)], [[np.zeros(3), np.ones(3)]]])
def test_autolabel_points_from_coords(kptcoords, special_points):
    path, dct = resolve_custom_points(kptcoords, {}, 0)
    assert path == 'Kpt0Kpt1'
    assert set(dct) == {'Kpt0', 'Kpt1'}