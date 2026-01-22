import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_lorentz(self):
    l_sy = np.array([0.29] * 18)
    l_sx = np.array([0.000972971, 0.000948268, 0.000707632, 0.000706679, 0.000706074, 0.000703918, 0.000698955, 0.000456856, 0.000455207, 0.000662717, 0.000654619, 0.000652694, 8.59202e-07, 0.00106589, 0.00106378, 0.00125483, 0.00140818, 0.00241839])
    l_dat = RealData([3.9094, 3.85945, 3.84976, 3.84716, 3.84551, 3.83964, 3.82608, 3.78847, 3.78163, 3.72558, 3.70274, 3.6973, 3.67373, 3.65982, 3.6562, 3.62498, 3.55525, 3.41886], [652, 910.5, 984, 1000, 1007.5, 1053, 1160.5, 1409.5, 1430, 1122, 957.5, 920, 777.5, 709.5, 698, 578.5, 418.5, 275.5], sx=l_sx, sy=l_sy)
    l_mod = Model(self.lorentz, meta=dict(name='Lorentz Peak'))
    l_odr = ODR(l_dat, l_mod, beta0=(1000.0, 0.1, 3.8))
    out = l_odr.run()
    assert_array_almost_equal(out.beta, np.array([1430.6780846149925, 0.1339050903453831, 3.779819360010901]))
    assert_array_almost_equal(out.sd_beta, np.array([0.7362118681133096, 0.0003506889994147165, 0.0002445120928140899]))
    assert_array_almost_equal(out.cov_beta, np.array([[0.24714409064597873, -6.906726191111084e-05, -3.123695327042499e-05], [-6.906726191111084e-05, 5.607753151733301e-08, 3.61332618327226e-08], [-3.123695327042499e-05, 3.61332618327226e-08, 2.726122002517173e-08]]))