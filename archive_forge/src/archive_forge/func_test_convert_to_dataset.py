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
def test_convert_to_dataset(self, draws, chains, data):
    dataset = convert_to_dataset(data.obj, group='posterior', coords={'school': np.arange(8)}, dims={'theta': ['school'], 'eta': ['school']})
    assert dataset.draw.shape == (draws,)
    assert dataset.chain.shape == (chains,)
    assert dataset.school.shape == (8,)
    assert dataset.theta.shape == (chains, draws, 8)