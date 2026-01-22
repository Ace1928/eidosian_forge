import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def test_NAAction_NA_types_numerical():
    for NA_types in [[], ['NaN'], ['None'], ['NaN', 'None']]:
        action = NAAction(NA_types=NA_types)
        for extra_shape in [(), (1,), (2,)]:
            arr = np.ones((4,) + extra_shape, dtype=float)
            nan_rows = [0, 2]
            if arr.ndim > 1 and arr.shape[1] > 1:
                arr[nan_rows, [0, 1]] = np.nan
            else:
                arr[nan_rows] = np.nan
            exp_NA_mask = np.zeros(4, dtype=bool)
            if 'NaN' in NA_types:
                exp_NA_mask[nan_rows] = True
            got_NA_mask = action.is_numerical_NA(arr)
            assert np.array_equal(got_NA_mask, exp_NA_mask)