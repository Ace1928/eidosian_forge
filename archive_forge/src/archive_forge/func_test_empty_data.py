import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_empty_data(self):
    beta0 = [0.02, 0.0]
    linear = Model(self.empty_data_func)
    empty_dat = Data([], [])
    assert_warns(OdrWarning, ODR, empty_dat, linear, beta0=beta0)
    empty_dat = RealData([], [])
    assert_warns(OdrWarning, ODR, empty_dat, linear, beta0=beta0)