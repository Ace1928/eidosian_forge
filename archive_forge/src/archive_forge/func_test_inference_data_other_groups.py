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
def test_inference_data_other_groups():
    datadict = {'a': np.random.randn(100), 'b': np.random.randn(1, 100, 10)}
    dataset = convert_to_dataset(datadict, coords={'c': np.arange(10)}, dims={'b': ['c']})
    with pytest.warns(UserWarning, match='not.+in.+InferenceData scheme'):
        idata = InferenceData(other_group=dataset)
    fails = check_multiple_attrs({'other_group': ['a', 'b']}, idata)
    assert not fails