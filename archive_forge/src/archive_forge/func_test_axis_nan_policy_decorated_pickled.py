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
@pytest.mark.parametrize(('hypotest', 'args', 'kwds', 'n_samples', 'n_outputs', 'paired', 'unpacker'), axis_nan_policy_cases)
def test_axis_nan_policy_decorated_pickled(hypotest, args, kwds, n_samples, n_outputs, paired, unpacker):
    if 'ttest_ci' in hypotest.__name__:
        pytest.skip("Can't pickle functions defined within functions.")
    rng = np.random.default_rng(0)
    if not unpacker:

        def unpacker(res):
            return res
    data = rng.uniform(size=(n_samples, 2, 30))
    pickled_hypotest = pickle.dumps(hypotest)
    unpickled_hypotest = pickle.loads(pickled_hypotest)
    res1 = unpacker(hypotest(*data, *args, axis=-1, **kwds))
    res2 = unpacker(unpickled_hypotest(*data, *args, axis=-1, **kwds))
    assert_allclose(res1, res2, rtol=1e-12)