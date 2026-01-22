from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize(('hypotest', 'args', 'kwds', 'n_samples', 'n_outputs', 'paired', 'unpacker'), axis_nan_policy_cases)
@pytest.mark.parametrize('nan_policy', ('propagate', 'omit', 'raise'))
@pytest.mark.parametrize('axis', (1,))
@pytest.mark.parametrize('data_generator', ('mixed',))
def test_axis_nan_policy_fast(hypotest, args, kwds, n_samples, n_outputs, paired, unpacker, nan_policy, axis, data_generator):
    _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired, unpacker, nan_policy, axis, data_generator)