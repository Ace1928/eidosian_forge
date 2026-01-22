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
def test_extend_errors_warnings(self, data_random):
    idata = data_random
    idata2 = create_data_random(groups=['prior', 'prior_predictive', 'observed_data'], seed=7)
    with pytest.raises(ValueError, match='Extending.+InferenceData objects only.'):
        idata.extend('something')
    with pytest.raises(ValueError, match='join must be either'):
        idata.extend(idata2, join='outer')
    idata2.add_groups(new_group=idata2.prior)
    with pytest.warns(UserWarning):
        idata.extend(idata2)