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
def test_group_names_invalid_args(self):
    ds = dict_to_dataset({'a': np.random.normal(size=(3, 10))})
    idata = InferenceData(posterior=(ds, ds))
    msg = "^\\'filter_groups\\' can only be None, \\'like\\', or \\'regex\\', got: 'foo'$"
    with pytest.raises(ValueError, match=msg):
        idata._group_names(('posterior',), filter_groups='foo')