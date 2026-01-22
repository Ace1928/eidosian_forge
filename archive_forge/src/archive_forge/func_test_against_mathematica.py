import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
from scipy.special._testutils import FuncData
def test_against_mathematica(self):
    points = np.array([[-7.89, 45.06, 6.66, 0.007792107366038881], [-0.05, 7.98, 24.13, 0.012068223646769913], [-13.98, 16.83, 42.37, 0.006244223636213236], [-12.66, 0.21, 6.32, 0.010052516161087379], [11.34, 4.25, 21.96, 0.011369892362727892], [-11.56, 20.4, 30.53, 0.007633276043209746], [-9.17, 25.61, 8.32, 0.011646345779083005], [16.59, 18.05, 2.5, 0.01363776883752681], [9.11, 2.12, 39.33, 0.007664404080727768], [-43.33, 0.3, 45.68, 0.003668046387533015]])
    FuncData(sc.voigt_profile, points, (0, 1, 2), 3, atol=0, rtol=1e-15).check()