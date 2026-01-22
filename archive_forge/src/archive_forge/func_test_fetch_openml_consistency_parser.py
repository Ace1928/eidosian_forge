import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
@fails_if_pypy
@pytest.mark.parametrize('data_id', [61, 1119, 40945])
def test_fetch_openml_consistency_parser(monkeypatch, data_id):
    """Check the consistency of the LIAC-ARFF and pandas parsers."""
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch_liac = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser='liac-arff')
    bunch_pandas = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser='pandas')
    data_liac, data_pandas = (bunch_liac.data, bunch_pandas.data)

    def convert_numerical_dtypes(series):
        pandas_series = data_pandas[series.name]
        if pd.api.types.is_numeric_dtype(pandas_series):
            return series.astype(pandas_series.dtype)
        else:
            return series
    data_liac_with_fixed_dtypes = data_liac.apply(convert_numerical_dtypes)
    pd.testing.assert_frame_equal(data_liac_with_fixed_dtypes, data_pandas)
    frame_liac, frame_pandas = (bunch_liac.frame, bunch_pandas.frame)
    pd.testing.assert_frame_equal(frame_pandas[bunch_pandas.feature_names], data_pandas)

    def convert_numerical_and_categorical_dtypes(series):
        pandas_series = frame_pandas[series.name]
        if pd.api.types.is_numeric_dtype(pandas_series):
            return series.astype(pandas_series.dtype)
        elif isinstance(pandas_series.dtype, pd.CategoricalDtype):
            return series.cat.rename_categories(pandas_series.cat.categories)
        else:
            return series
    frame_liac_with_fixed_dtypes = frame_liac.apply(convert_numerical_and_categorical_dtypes)
    pd.testing.assert_frame_equal(frame_liac_with_fixed_dtypes, frame_pandas)