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
def test_io_function(self, data, eight_schools_params):
    inference_data = self.get_inference_data(data, eight_schools_params)
    test_dict = {'posterior': ['eta', 'theta', 'mu', 'tau'], 'posterior_predictive': ['eta', 'theta', 'mu', 'tau'], 'sample_stats': ['eta', 'theta', 'mu', 'tau'], 'prior': ['eta', 'theta', 'mu', 'tau'], 'prior_predictive': ['eta', 'theta', 'mu', 'tau'], 'sample_stats_prior': ['eta', 'theta', 'mu', 'tau'], 'observed_data': ['J', 'y', 'sigma']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, '..', 'saved_models')
    filepath = os.path.join(data_directory, 'io_function_testfile.nc')
    to_netcdf(inference_data, filepath)
    assert os.path.exists(filepath)
    assert os.path.getsize(filepath) > 0
    inference_data2 = from_netcdf(filepath)
    fails = check_multiple_attrs(test_dict, inference_data2)
    assert not fails
    os.remove(filepath)
    assert not os.path.exists(filepath)