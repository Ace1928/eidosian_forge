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
def test_nd_to_inference_data(self):
    shape = (1, 2, 3, 4, 5)
    inference_data = convert_to_inference_data(xr.DataArray(np.random.randn(*shape), dims=('chain', 'draw', 'dim_0', 'dim_1', 'dim_2')), group='prior')
    var_name = list(inference_data.prior.data_vars)[0]
    assert hasattr(inference_data, 'prior')
    assert len(inference_data.prior.data_vars) == 1
    assert inference_data.prior.chain.shape == shape[:1]
    assert inference_data.prior.draw.shape == shape[1:2]
    assert inference_data.prior[var_name].shape == shape