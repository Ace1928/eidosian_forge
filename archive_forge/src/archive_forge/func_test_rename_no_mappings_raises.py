from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_no_mappings_raises(self):
    df = DataFrame([[1]])
    msg = 'must pass an index to rename'
    with pytest.raises(TypeError, match=msg):
        df.rename()
    with pytest.raises(TypeError, match=msg):
        df.rename(None, index=None)
    with pytest.raises(TypeError, match=msg):
        df.rename(None, columns=None)
    with pytest.raises(TypeError, match=msg):
        df.rename(None, columns=None, index=None)