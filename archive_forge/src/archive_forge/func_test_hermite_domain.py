import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
def test_hermite_domain():
    assert np.isnan(_ufuncs.eval_hermite(-1, 1.0))
    assert np.isnan(_ufuncs.eval_hermitenorm(-1, 1.0))