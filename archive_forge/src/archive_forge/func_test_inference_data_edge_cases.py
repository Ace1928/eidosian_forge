import importlib
import os
from collections import namedtuple
from copy import deepcopy
from html import escape
from typing import Dict
from tempfile import TemporaryDirectory
from urllib.parse import urlunsplit
import numpy as np
import pytest
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.testing import assert_identical
from ... import (
from ...data.base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs
from ...data.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata
from ..helpers import (  # pylint: disable=unused-import
def test_inference_data_edge_cases(self):
    log_likelihood = {'y': np.random.randn(4, 100), 'log_likelihood': np.random.randn(4, 100, 8)}
    with pytest.warns(UserWarning, match='log_likelihood.+in posterior'):
        assert from_dict(posterior=log_likelihood) is not None
    assert from_dict(observed_data=log_likelihood, dims=None) is not None