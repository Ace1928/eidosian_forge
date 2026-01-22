import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_named_results(res, attributes, ma=False):
    for i, attr in enumerate(attributes):
        if ma:
            ma_npt.assert_equal(res[i], getattr(res, attr))
        else:
            npt.assert_equal(res[i], getattr(res, attr))