import numpy as np
import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_inplace_mutation_resets_values():
    levels = [['a', 'b', 'c'], [4]]
    levels2 = [[1, 2, 3], ['a']]
    codes = [[0, 1, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]]
    mi1 = MultiIndex(levels=levels, codes=codes)
    mi2 = MultiIndex(levels=levels2, codes=codes)
    assert '_values' not in mi1._cache
    assert '_values' not in mi2._cache
    vals = mi1.values.copy()
    vals2 = mi2.values.copy()
    assert mi1._values is mi1._cache['_values']
    assert mi1.values is mi1._cache['_values']
    assert isinstance(mi1._cache['_values'], np.ndarray)
    new_vals = mi1.set_levels(levels2).values
    tm.assert_almost_equal(vals2, new_vals)
    tm.assert_almost_equal(mi1._cache['_values'], vals)
    tm.assert_almost_equal(mi1.values, vals)
    codes2 = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    exp_values = np.empty((6,), dtype=object)
    exp_values[:] = [(1, 'a')] * 6
    assert exp_values.shape == (6,)
    new_mi = mi2.set_codes(codes2)
    assert '_values' not in new_mi._cache
    new_values = new_mi.values
    assert '_values' in new_mi._cache
    tm.assert_almost_equal(mi2._cache['_values'], vals2)
    tm.assert_almost_equal(exp_values, new_values)