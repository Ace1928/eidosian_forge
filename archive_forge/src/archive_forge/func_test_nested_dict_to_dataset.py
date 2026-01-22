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
def test_nested_dict_to_dataset():
    datadict = {'top': {'a': np.random.randn(100), 'b': np.random.randn(1, 100, 10)}, 'd': np.random.randn(100)}
    dataset = convert_to_dataset(datadict, coords={'c': np.arange(10)}, dims={('top', 'b'): ['c']})
    assert set(dataset.data_vars) == {('top', 'a'), ('top', 'b'), 'd'}
    assert set(dataset.coords) == {'chain', 'draw', 'c'}
    assert set(dataset['top', 'a'].coords) == {'chain', 'draw'}
    assert set(dataset['top', 'b'].coords) == {'chain', 'draw', 'c'}
    assert set(dataset.d.coords) == {'chain', 'draw'}