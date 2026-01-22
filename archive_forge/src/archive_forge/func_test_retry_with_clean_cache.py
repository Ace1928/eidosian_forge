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
def test_retry_with_clean_cache(tmpdir):
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    location = _get_local_path(openml_path, cache_directory)
    os.makedirs(os.path.dirname(location))
    with open(location, 'w') as f:
        f.write('')

    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        if os.path.exists(location):
            raise Exception('File exist!')
        return 1
    warn_msg = 'Invalid cache, redownloading file'
    with pytest.warns(RuntimeWarning, match=warn_msg):
        result = _load_data()
    assert result == 1