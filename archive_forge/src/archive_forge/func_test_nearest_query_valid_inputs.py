import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_nearest_query_valid_inputs(self):
    nd = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 2]])
    NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])
    with assert_raises(TypeError):
        NI([0.5, 0.5], query_options='not a dictionary')