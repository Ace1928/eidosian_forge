import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_data_load():
    rcParams['data.load'] = 'lazy'
    idata_lazy = load_arviz_data('centered_eight')
    assert isinstance(idata_lazy.posterior.mu.variable._data, MemoryCachedArray)
    assert rcParams['data.load'] == 'lazy'
    rcParams['data.load'] = 'eager'
    idata_eager = load_arviz_data('centered_eight')
    assert isinstance(idata_eager.posterior.mu.variable._data, np.ndarray)
    assert rcParams['data.load'] == 'eager'