import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('batch', (-1, 0))
def test_batch_generator_iv(self, batch):
    with pytest.raises(ValueError, match='`batch` must be positive.'):
        list(_resampling._batch_generator([1, 2, 3], batch))