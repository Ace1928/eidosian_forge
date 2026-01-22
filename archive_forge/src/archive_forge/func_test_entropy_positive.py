import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_entropy_positive(self):
    pk = [0.5, 0.2, 0.3]
    qk = [0.1, 0.25, 0.65]
    eself = stats.entropy(pk, pk)
    edouble = stats.entropy(pk, qk)
    assert_(0.0 == eself)
    assert_(edouble >= 0.0)