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
def test_open_openml_url_retry_on_network_error(monkeypatch):

    def _mock_urlopen_network_error(request, *args, **kwargs):
        raise HTTPError(url=None, code=404, msg='Simulated network error', hdrs=None, fp=BytesIO())
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen_network_error)
    invalid_openml_url = 'invalid-url'
    with pytest.warns(UserWarning, match=re.escape(f'A network error occurred while downloading {_OPENML_PREFIX + invalid_openml_url}. Retrying...')) as record:
        with pytest.raises(HTTPError, match='Simulated network error'):
            _open_openml_url(invalid_openml_url, None, delay=0)
        assert len(record) == 3