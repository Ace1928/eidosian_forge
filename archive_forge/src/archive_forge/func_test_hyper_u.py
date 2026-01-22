import math
import pytest
from mpmath import *
def test_hyper_u():
    mp.dps = 15
    assert hyperu(2, -3, 0).ae(0.05)
    assert hyperu(2, -3.5, 0).ae(4.0 / 99)
    assert hyperu(2, 0, 0) == 0.5
    assert hyperu(-5, 1, 0) == -120
    assert hyperu(-5, 2, 0) == inf
    assert hyperu(-5, -2, 0) == 0
    assert hyperu(7, 7, 3).ae(0.00014681269365593504)
    assert hyperu(2, -3, 4).ae(0.011836478100271995)
    assert hyperu(3, 4, 5).ae(1.0 / 125)
    assert hyperu(2, 3, 0.0625) == 256
    assert hyperu(-1, 2, 0.25 + 0.5j) == -1.75 + 0.5j
    assert hyperu(0.5, 1.5, 7.25).ae(2 / sqrt(29))
    assert hyperu(2, 6, pi).ae(0.558044398259134)
    assert (hyperu((3, 2), 8, 100 + 201j) * 10 ** 4).ae(-0.3797318333856739 - 2.9974928453561707j)
    assert (hyperu((5, 2), (-1, 2), -5000) * 10 ** 10).ae(-5.668187792688166j)