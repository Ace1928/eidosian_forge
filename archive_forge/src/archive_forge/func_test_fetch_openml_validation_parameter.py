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
@pytest.mark.parametrize('params, err_msg', [({'parser': 'unknown'}, "The 'parser' parameter of fetch_openml must be a str among"), ({'as_frame': 'unknown'}, "The 'as_frame' parameter of fetch_openml must be an instance")])
def test_fetch_openml_validation_parameter(monkeypatch, params, err_msg):
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    with pytest.raises(ValueError, match=err_msg):
        fetch_openml(data_id=data_id, **params)