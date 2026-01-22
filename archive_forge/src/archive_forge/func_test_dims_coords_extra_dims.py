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
def test_dims_coords_extra_dims():
    shape = (4, 20)
    var_name = 'x'
    with pytest.warns(UserWarning):
        dims, coords = generate_dims_coords(shape, var_name, dims=['xx', 'xy', 'xz'])
    assert 'xx' in dims
    assert 'xy' in dims
    assert 'xz' in dims
    assert len(coords['xx']) == 4
    assert len(coords['xy']) == 20