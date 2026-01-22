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
def test_use_all(self, data):
    dataset = convert_to_dataset(data.datadict, coords=data.coords, dims=data.dims)
    assert set(dataset.data_vars) == {'a', 'b', 'c'}
    assert set(dataset.coords) == {'chain', 'draw', 'c1', 'c2', 'b1'}
    assert set(dataset.a.coords) == {'chain', 'draw'}
    assert set(dataset.b.coords) == {'chain', 'draw', 'b1'}
    assert set(dataset.c.coords) == {'chain', 'draw', 'c1', 'c2'}