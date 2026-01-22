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
def test_nd_to_dataset(self):
    shape = (1, 2, 3, 4, 5)
    dataset = convert_to_dataset(xr.DataArray(np.random.randn(*shape), dims=('chain', 'draw', 'dim_0', 'dim_1', 'dim_2')))
    var_name = list(dataset.data_vars)[0]
    assert len(dataset.data_vars) == 1
    assert dataset.chain.shape == shape[:1]
    assert dataset.draw.shape == shape[1:2]
    assert dataset[var_name].shape == shape