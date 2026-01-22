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
def test_fetch_openml_difference_parsers(monkeypatch):
    """Check the difference between liac-arff and pandas parser."""
    pytest.importorskip('pandas')
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    as_frame = False
    bunch_liac_arff = fetch_openml(data_id=data_id, as_frame=as_frame, cache=False, parser='liac-arff')
    bunch_pandas = fetch_openml(data_id=data_id, as_frame=as_frame, cache=False, parser='pandas')
    assert bunch_liac_arff.data.dtype.kind == 'f'
    assert bunch_pandas.data.dtype == 'O'