from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_regex_replace_dict_mixed(self, mix_abc):
    dfmix = DataFrame(mix_abc)
    res = dfmix.replace({'b': '\\s*\\.\\s*'}, {'b': np.nan}, regex=True)
    res2 = dfmix.copy()
    return_value = res2.replace({'b': '\\s*\\.\\s*'}, {'b': np.nan}, inplace=True, regex=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', np.nan, np.nan], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    res = dfmix.replace({'b': '\\s*(\\.)\\s*'}, {'b': '\\1ty'}, regex=True)
    res2 = dfmix.copy()
    return_value = res2.replace({'b': '\\s*(\\.)\\s*'}, {'b': '\\1ty'}, inplace=True, regex=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', '.ty', '.ty'], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    res = dfmix.replace(regex={'b': '\\s*(\\.)\\s*'}, value={'b': '\\1ty'})
    res2 = dfmix.copy()
    return_value = res2.replace(regex={'b': '\\s*(\\.)\\s*'}, value={'b': '\\1ty'}, inplace=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', '.ty', '.ty'], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    expec = DataFrame({'a': mix_abc['a'], 'b': [np.nan, 'b', '.', '.'], 'c': mix_abc['c']})
    res = dfmix.replace('a', {'b': np.nan}, regex=True)
    res2 = dfmix.copy()
    return_value = res2.replace('a', {'b': np.nan}, regex=True, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    res = dfmix.replace('a', {'b': np.nan}, regex=True)
    res2 = dfmix.copy()
    return_value = res2.replace(regex='a', value={'b': np.nan}, inplace=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': [np.nan, 'b', '.', '.'], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)