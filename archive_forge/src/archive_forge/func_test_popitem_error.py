import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_popitem_error():
    """Check rcParams popitem error."""
    with pytest.raises(TypeError, match='keys cannot be deleted.*get\\(key\\)'):
        rcParams.popitem()