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
def test_1d_dataset(self):
    size = 100
    dataset = convert_to_dataset(xr.DataArray(np.random.randn(1, size), name='plot', dims=('chain', 'draw')))
    assert len(dataset.data_vars) == 1
    assert 'plot' in dataset.data_vars
    assert dataset.chain.shape == (1,)
    assert dataset.draw.shape == (size,)