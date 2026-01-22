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
def test_more_chains_than_draws(self):
    shape = (10, 4)
    with pytest.warns(UserWarning):
        inference_data = convert_to_inference_data(np.random.randn(*shape), group='prior')
    assert hasattr(inference_data, 'prior')
    assert len(inference_data.prior.data_vars) == 1
    var_name = list(inference_data.prior.data_vars)[0]
    assert len(inference_data.prior.coords) == len(shape)
    assert inference_data.prior.chain.shape == shape[:1]
    assert inference_data.prior.draw.shape == shape[1:2]
    assert inference_data.prior[var_name].shape == shape