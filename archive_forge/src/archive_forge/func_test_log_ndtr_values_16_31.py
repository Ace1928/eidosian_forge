import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_log_ndtr_values_16_31(self):
    x = np.array([16.15, 20.3, 21.4, 26.2, 30.9])
    expected = [-5.678084565148492e-59, -6.429244467698346e-92, -6.680402412553295e-102, -1.328698078458869e-151, -5.972288641838264e-210]
    y = sc.log_ndtr(x)
    assert_allclose(y, expected, rtol=2e-13)