import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
def test_series_cat_api_equality():
    modin_dir = [obj for obj in dir(pd.Series.cat) if obj[0] != '_']
    pandas_dir = [obj for obj in dir(pandas.Series.cat) if obj[0] != '_']
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin), 'Differences found in API: {}'.format(len(missing_from_modin))
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), 'Differences found in API: {}'.format(extra_in_modin)
    assert_parameters_eq((pandas.core.arrays.Categorical, pd.Series.cat), modin_dir, [])