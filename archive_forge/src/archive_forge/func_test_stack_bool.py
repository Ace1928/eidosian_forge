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
def test_stack_bool(self):
    datadict = {'a': np.random.randn(100), 'b': np.random.randn(1, 100, 10), 'c': np.random.randn(1, 100, 3, 4)}
    coords = {'c1': np.arange(3), 'c99': np.arange(4), 'b1': np.arange(10)}
    dims = {'c': ['c1', 'c99'], 'b': ['b1']}
    dataset = from_dict(posterior=datadict, coords=coords, dims=dims)
    assert_identical(dataset.stack(z=['c1', 'c99'], create_index=False).posterior, dataset.posterior.stack(z=['c1', 'c99'], create_index=False))