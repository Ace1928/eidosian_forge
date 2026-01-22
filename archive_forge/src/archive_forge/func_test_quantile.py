import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'nearest', 'midpoint'])
@pytest.mark.parametrize('a_vals,b_vals', [([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), ([1, 2, 3, 4], [4, 3, 2, 1]), ([1, 2, 3, 4, 5], [4, 3, 2, 1]), ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]), ([1.0, np.nan, 3.0, np.nan, 5.0], [5.0, np.nan, 3.0, np.nan, 1.0]), ([np.nan, 4.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 2.0, np.nan]), (pd.date_range('1/1/18', freq='D', periods=5), pd.date_range('1/1/18', freq='D', periods=5)[::-1]), (pd.date_range('1/1/18', freq='D', periods=5).as_unit('s'), pd.date_range('1/1/18', freq='D', periods=5)[::-1].as_unit('s')), ([np.nan] * 5, [np.nan] * 5)])
@pytest.mark.parametrize('q', [0, 0.25, 0.5, 0.75, 1])
def test_quantile(interpolation, a_vals, b_vals, q, request):
    if interpolation == 'nearest' and q == 0.5 and isinstance(b_vals, list) and (b_vals == [4, 3, 2, 1]):
        request.node.add_marker(pytest.mark.xfail(reason='Unclear numpy expectation for nearest result with equidistant data'))
    all_vals = pd.concat([pd.Series(a_vals), pd.Series(b_vals)])
    a_expected = pd.Series(a_vals).quantile(q, interpolation=interpolation)
    b_expected = pd.Series(b_vals).quantile(q, interpolation=interpolation)
    df = DataFrame({'key': ['a'] * len(a_vals) + ['b'] * len(b_vals), 'val': all_vals})
    expected = DataFrame([a_expected, b_expected], columns=['val'], index=Index(['a', 'b'], name='key'))
    if all_vals.dtype.kind == 'M' and expected.dtypes.values[0].kind == 'M':
        expected = expected.astype(all_vals.dtype)
    result = df.groupby('key').quantile(q, interpolation=interpolation)
    tm.assert_frame_equal(result, expected)