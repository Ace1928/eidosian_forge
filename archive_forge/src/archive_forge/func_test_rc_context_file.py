import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_rc_context_file():
    path = os.path.dirname(os.path.abspath(__file__))
    rcParams['data.load'] = 'lazy'
    with rc_context(fname=os.path.join(path, '../test.rcparams')):
        assert rcParams['data.load'] == 'eager'
    assert rcParams['data.load'] == 'lazy'