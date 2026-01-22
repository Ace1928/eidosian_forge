import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_private_entropy(distfn, args, superclass):
    npt.assert_allclose(distfn._entropy(*args), superclass._entropy(distfn, *args))