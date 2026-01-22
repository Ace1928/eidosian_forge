from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_prediction_increment_pandas_dates_daily():
    endog = dta[2].copy()
    endog.index = date_indexes[0][0]
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)
    start_key = 0
    end_key = 3
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[:4]), True)
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start='1950-01-02', periods=5, freq='D')
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = '1950-01-02'
    end_key = '1950-01-04'
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 3)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[1:4]), True)
    start_key = '1950-01-01'
    end_key = '1950-01-08'
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start='1950-01-01', periods=8, freq='D')
    assert_equal(prediction_index.equals(desired_index), True)
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start='1950-01-01', periods=3, freq='D')
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    loc, index, index_was_expanded = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.date_range(start='1950-01-01', periods=3, freq='D')
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    loc, index, index_was_expanded = mod._get_index_label_loc('1950-01-03')
    assert_equal(loc, 2)
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)