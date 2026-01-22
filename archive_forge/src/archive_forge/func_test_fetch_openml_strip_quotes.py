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
def test_fetch_openml_strip_quotes(monkeypatch):
    """Check that we strip the single quotes when used as a string delimiter.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23381
    """
    pd = pytest.importorskip('pandas')
    data_id = 40966
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'as_frame': True, 'cache': False, 'data_id': data_id}
    mice_pandas = fetch_openml(parser='pandas', **common_params)
    mice_liac_arff = fetch_openml(parser='liac-arff', **common_params)
    pd.testing.assert_series_equal(mice_pandas.target, mice_liac_arff.target)
    assert not mice_pandas.target.str.startswith("'").any()
    assert not mice_pandas.target.str.endswith("'").any()
    mice_pandas = fetch_openml(parser='pandas', target_column='NUMB_N', **common_params)
    mice_liac_arff = fetch_openml(parser='liac-arff', target_column='NUMB_N', **common_params)
    pd.testing.assert_series_equal(mice_pandas.frame['class'], mice_liac_arff.frame['class'])
    assert not mice_pandas.frame['class'].str.startswith("'").any()
    assert not mice_pandas.frame['class'].str.endswith("'").any()