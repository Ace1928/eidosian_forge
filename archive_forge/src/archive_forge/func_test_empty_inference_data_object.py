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
def test_empty_inference_data_object(self):
    inference_data = InferenceData()
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, '..', 'saved_models')
    filepath = os.path.join(data_directory, 'empty_test_file.nc')
    assert not os.path.exists(filepath)
    inference_data.to_netcdf(filepath)
    assert os.path.exists(filepath)
    os.remove(filepath)
    assert not os.path.exists(filepath)