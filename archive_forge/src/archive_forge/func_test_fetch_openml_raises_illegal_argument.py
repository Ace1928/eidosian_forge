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
@pytest.mark.parametrize('params, err_type, err_msg', [({'data_id': -1, 'name': None, 'version': 'version'}, ValueError, "The 'version' parameter of fetch_openml must be an int in the range"), ({'data_id': -1, 'name': 'nAmE'}, ValueError, "The 'data_id' parameter of fetch_openml must be an int in the range"), ({'data_id': -1, 'name': 'nAmE', 'version': 'version'}, ValueError, "The 'version' parameter of fetch_openml must be an int"), ({}, ValueError, 'Neither name nor data_id are provided. Please provide name or data_id.')])
def test_fetch_openml_raises_illegal_argument(params, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        fetch_openml(**params)