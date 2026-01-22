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
@pytest.mark.parametrize('as_frame, parser', [(True, 'liac-arff'), (False, 'liac-arff'), (True, 'pandas'), (False, 'pandas')])
def test_fetch_openml_verify_checksum(monkeypatch, as_frame, cache, tmpdir, parser):
    """Check that the checksum is working as expected."""
    if as_frame or parser == 'pandas':
        pytest.importorskip('pandas')
    data_id = 2
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    original_data_module = OPENML_TEST_DATA_MODULE + '.' + f'id_{data_id}'
    original_data_file_name = 'data-v1-dl-1666876.arff.gz'
    original_data_path = resources.files(original_data_module) / original_data_file_name
    corrupt_copy_path = tmpdir / 'test_invalid_checksum.arff'
    with original_data_path.open('rb') as orig_file:
        orig_gzip = gzip.open(orig_file, 'rb')
        data = bytearray(orig_gzip.read())
        data[len(data) - 1] = 37
    with gzip.GzipFile(corrupt_copy_path, 'wb') as modified_gzip:
        modified_gzip.write(data)
    mocked_openml_url = sklearn.datasets._openml.urlopen

    def swap_file_mock(request, *args, **kwargs):
        url = request.get_full_url()
        if url.endswith('data/v1/download/1666876'):
            with open(corrupt_copy_path, 'rb') as f:
                corrupted_data = f.read()
            return _MockHTTPResponse(BytesIO(corrupted_data), is_gzip=True)
        else:
            return mocked_openml_url(request)
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', swap_file_mock)
    with pytest.raises(ValueError) as exc:
        sklearn.datasets.fetch_openml(data_id=data_id, cache=False, as_frame=as_frame, parser=parser)
    assert exc.match('1666876')