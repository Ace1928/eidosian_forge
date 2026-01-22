import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_clear_error():
    """Check that rcParams cannot be cleared."""
    with pytest.raises(TypeError, match='keys cannot be deleted'):
        rcParams.clear()