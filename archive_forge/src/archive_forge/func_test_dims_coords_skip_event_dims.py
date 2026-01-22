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
@pytest.mark.parametrize('shape', [(4, 20), (4, 20, 1)])
def test_dims_coords_skip_event_dims(shape):
    coords = {'x': np.arange(4), 'y': np.arange(20), 'z': np.arange(5)}
    dims, coords = generate_dims_coords(shape, 'name', dims=['x', 'y', 'z'], coords=coords, skip_event_dims=True)
    assert 'x' in dims
    assert 'y' in dims
    assert 'z' not in dims
    assert len(coords['x']) == 4
    assert len(coords['y']) == 20
    assert 'z' not in coords