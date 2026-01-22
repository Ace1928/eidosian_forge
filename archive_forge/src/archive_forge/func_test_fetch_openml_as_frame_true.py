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
@pytest.mark.parametrize('data_id, dataset_params, n_samples, n_features, n_targets', [(61, {'data_id': 61}, 150, 4, 1), (61, {'name': 'iris', 'version': 1}, 150, 4, 1), (2, {'data_id': 2}, 11, 38, 1), (2, {'name': 'anneal', 'version': 1}, 11, 38, 1), (561, {'data_id': 561}, 209, 7, 1), (561, {'name': 'cpu', 'version': 1}, 209, 7, 1), (40589, {'data_id': 40589}, 13, 72, 6), (1119, {'data_id': 1119}, 10, 14, 1), (1119, {'name': 'adult-census'}, 10, 14, 1), (40966, {'data_id': 40966}, 7, 77, 1), (40966, {'name': 'MiceProtein'}, 7, 77, 1), (40945, {'data_id': 40945}, 1309, 13, 1)])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_as_frame_true(monkeypatch, data_id, dataset_params, n_samples, n_features, n_targets, parser, gzip_response):
    """Check the behaviour of `fetch_openml` with `as_frame=True`.

    Fetch by ID and/or name (depending if the file was previously cached).
    """
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    bunch = fetch_openml(as_frame=True, cache=False, parser=parser, **dataset_params)
    assert int(bunch.details['id']) == data_id
    assert isinstance(bunch, Bunch)
    assert isinstance(bunch.frame, pd.DataFrame)
    assert bunch.frame.shape == (n_samples, n_features + n_targets)
    assert isinstance(bunch.data, pd.DataFrame)
    assert bunch.data.shape == (n_samples, n_features)
    if n_targets == 1:
        assert isinstance(bunch.target, pd.Series)
        assert bunch.target.shape == (n_samples,)
    else:
        assert isinstance(bunch.target, pd.DataFrame)
        assert bunch.target.shape == (n_samples, n_targets)
    assert bunch.categories is None