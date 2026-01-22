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
@pytest.mark.parametrize('data_id', [61, 561, 40589, 1119])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_equivalence_array_return_X_y(monkeypatch, data_id, parser):
    """Check the behaviour of `return_X_y=True` when `as_frame=False`."""
    pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch = fetch_openml(data_id=data_id, as_frame=False, cache=False, return_X_y=False, parser=parser)
    X, y = fetch_openml(data_id=data_id, as_frame=False, cache=False, return_X_y=True, parser=parser)
    assert_array_equal(bunch.data, X)
    assert_array_equal(bunch.target, y)