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
@pytest.mark.parametrize('nan_policy', ('omit', 'propagate'))
@pytest.mark.parametrize(('hypotest', 'args', 'kwds', 'n_samples', 'unpacker'), ((stats.gmean, tuple(), dict(), 1, lambda x: (x,)), (stats.mannwhitneyu, tuple(), {'method': 'asymptotic'}, 2, None)))
@pytest.mark.parametrize(('sample_shape', 'axis_cases'), (((2, 3, 3, 4), (None, 0, -1, (0, 2), (1, -1), (3, 1, 2, 0))), ((10,), (0, -1)), ((20, 0), (0, 1))))
def test_keepdims(hypotest, args, kwds, n_samples, unpacker, sample_shape, axis_cases, nan_policy):
    if not unpacker:

        def unpacker(res):
            return res
    rng = np.random.default_rng(0)
    data = [rng.random(sample_shape) for _ in range(n_samples)]
    nan_data = [sample.copy() for sample in data]
    nan_mask = [rng.random(sample_shape) < 0.2 for _ in range(n_samples)]
    for sample, mask in zip(nan_data, nan_mask):
        sample[mask] = np.nan
    for axis in axis_cases:
        expected_shape = list(sample_shape)
        if axis is None:
            expected_shape = np.ones(len(sample_shape))
        elif isinstance(axis, int):
            expected_shape[axis] = 1
        else:
            for ax in axis:
                expected_shape[ax] = 1
        expected_shape = tuple(expected_shape)
        res = unpacker(hypotest(*data, *args, axis=axis, keepdims=True, **kwds))
        res_base = unpacker(hypotest(*data, *args, axis=axis, keepdims=False, **kwds))
        nan_res = unpacker(hypotest(*nan_data, *args, axis=axis, keepdims=True, nan_policy=nan_policy, **kwds))
        nan_res_base = unpacker(hypotest(*nan_data, *args, axis=axis, keepdims=False, nan_policy=nan_policy, **kwds))
        for r, r_base, rn, rn_base in zip(res, res_base, nan_res, nan_res_base):
            assert r.shape == expected_shape
            r = np.squeeze(r, axis=axis)
            assert_equal(r, r_base)
            assert rn.shape == expected_shape
            rn = np.squeeze(rn, axis=axis)
            assert_equal(rn, rn_base)