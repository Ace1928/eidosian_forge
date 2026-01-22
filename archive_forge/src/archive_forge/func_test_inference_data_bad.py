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
def test_inference_data_bad(self):
    x = np.random.randn(4, 100)
    with pytest.raises(TypeError):
        from_dict(posterior=x)
    with pytest.raises(TypeError):
        from_dict(posterior_predictive=x)
    with pytest.raises(TypeError):
        from_dict(sample_stats=x)
    with pytest.raises(TypeError):
        from_dict(prior=x)
    with pytest.raises(TypeError):
        from_dict(prior_predictive=x)
    with pytest.raises(TypeError):
        from_dict(sample_stats_prior=x)
    with pytest.raises(TypeError):
        from_dict(observed_data=x)