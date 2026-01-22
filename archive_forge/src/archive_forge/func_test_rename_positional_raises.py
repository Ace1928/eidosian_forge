from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_positional_raises(self):
    df = DataFrame(columns=['A', 'B'])
    msg = 'rename\\(\\) takes from 1 to 2 positional arguments'
    with pytest.raises(TypeError, match=msg):
        df.rename(None, str.lower)