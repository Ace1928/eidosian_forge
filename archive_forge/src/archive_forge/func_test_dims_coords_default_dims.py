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
@pytest.mark.parametrize('in_dims', (['dim1', 'dim2'], ['draw', 'dim1', 'dim2'], ['chain', 'draw', 'dim1', 'dim2']))
def test_dims_coords_default_dims(in_dims):
    shape = (4, 7)
    var_name = 'x'
    dims, coords = generate_dims_coords(shape, var_name, dims=in_dims, coords={'chain': ['a', 'b', 'c']}, default_dims=['chain', 'draw'])
    assert 'dim1' in dims
    assert 'dim2' in dims
    assert ('chain' in dims) == ('chain' in in_dims)
    assert ('draw' in dims) == ('draw' in in_dims)
    assert len(coords['dim1']) == 4
    assert len(coords['dim2']) == 7
    assert len(coords['chain']) == 3
    assert 'draw' not in coords