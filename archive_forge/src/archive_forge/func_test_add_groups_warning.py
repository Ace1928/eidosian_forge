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
def test_add_groups_warning(self, data_random):
    data = np.random.normal(size=(4, 500, 8))
    idata = data_random
    with pytest.warns(UserWarning, match='The group.+not defined in the InferenceData scheme'):
        idata.add_groups({'new_group': idata.posterior})
    with pytest.warns(UserWarning, match='the default dims.+will be added automatically'):
        idata.add_groups(constant_data={'a': data[..., 0], 'b': data})
    assert idata.new_group.equals(idata.posterior)