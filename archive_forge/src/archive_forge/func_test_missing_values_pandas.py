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
@pytest.mark.parametrize('gzip_response', [True, False])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_missing_values_pandas(monkeypatch, gzip_response, parser):
    """check that missing values in categories are compatible with pandas
    categorical"""
    pytest.importorskip('pandas')
    data_id = 42585
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    penguins = fetch_openml(data_id=data_id, cache=False, as_frame=True, parser=parser)
    cat_dtype = penguins.data.dtypes['sex']
    assert penguins.data['sex'].isna().any()
    assert_array_equal(cat_dtype.categories, ['FEMALE', 'MALE', '_'])