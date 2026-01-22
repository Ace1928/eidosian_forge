import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_rcparams_repr_str():
    """Check both repr and str print all keys."""
    repr_str = repr(rcParams)
    str_str = str(rcParams)
    assert repr_str.startswith('RcParams')
    for string in (repr_str, str_str):
        assert all((key in string for key in rcParams.keys()))