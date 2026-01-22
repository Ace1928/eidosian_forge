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
@pytest.mark.parametrize('args_res', (([('posterior', 'sample_stats')], ('posterior', 'sample_stats')), (['posterior', 'like'], ('posterior', 'warmup_posterior', 'posterior_predictive')), (['^posterior', 'regex'], ('posterior', 'posterior_predictive')), ([('~^warmup', '~^obs'), 'regex'], ('posterior', 'sample_stats', 'posterior_predictive')), (['~observed_vars'], ('posterior', 'sample_stats', 'warmup_posterior', 'warmup_sample_stats'))))
def test_group_names(self, args_res):
    args, result = args_res
    ds = dict_to_dataset({'a': np.random.normal(size=(3, 10))})
    idata = InferenceData(posterior=(ds, ds), sample_stats=(ds, ds), observed_data=ds, posterior_predictive=ds)
    group_names = idata._group_names(*args)
    assert np.all([name in result for name in group_names])