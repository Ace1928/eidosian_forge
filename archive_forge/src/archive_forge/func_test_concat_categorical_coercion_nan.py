import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_coercion_nan(self):
    s1 = Series(np.array([np.nan, np.nan], dtype=np.float64), dtype='category')
    s2 = Series([np.nan, 1])
    exp = Series([np.nan, np.nan, np.nan, 1])
    tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
    s1 = Series([1, np.nan], dtype='category')
    s2 = Series([np.nan, np.nan])
    exp = Series([1, np.nan, np.nan, np.nan], dtype='float')
    tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
    s1 = Series([np.nan, np.nan], dtype='category')
    s2 = Series([np.nan, np.nan])
    exp = Series([np.nan, np.nan, np.nan, np.nan])
    tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
    tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
    tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)
    s1 = Series([np.nan, np.nan], dtype='category')
    s2 = Series([np.nan, np.nan], dtype='category')
    exp = Series([np.nan, np.nan, np.nan, np.nan], dtype='category')
    tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)