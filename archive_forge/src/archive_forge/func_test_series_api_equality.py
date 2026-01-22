import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
def test_series_api_equality():
    modin_dir = [obj for obj in dir(pd.Series) if obj[0] != '_']
    pandas_dir = [obj for obj in dir(pandas.Series) if obj[0] != '_']
    ignore = ['timetuple']
    missing_from_modin = set(pandas_dir) - set(modin_dir) - set(ignore)
    assert not len(missing_from_modin), 'Differences found in API: {}'.format(missing_from_modin)
    ignore_in_modin = ['modin']
    extra_in_modin = set(modin_dir) - set(ignore_in_modin) - set(pandas_dir)
    assert not len(extra_in_modin), 'Differences found in API: {}'.format(extra_in_modin)
    allowed_different = ['modin']
    assert_parameters_eq((pandas.Series, pd.Series), modin_dir, allowed_different)