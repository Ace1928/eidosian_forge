from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_axis_style_raises(self):
    df = DataFrame({'A': [1, 2], 'B': [1, 2]}, index=['0', '1'])
    over_spec_msg = "Cannot specify both 'axis' and any of 'index' or 'columns'"
    with pytest.raises(TypeError, match=over_spec_msg):
        df.rename(index=str.lower, axis=1)
    with pytest.raises(TypeError, match=over_spec_msg):
        df.rename(index=str.lower, axis='columns')
    with pytest.raises(TypeError, match=over_spec_msg):
        df.rename(columns=str.lower, axis='columns')
    with pytest.raises(TypeError, match=over_spec_msg):
        df.rename(index=str.lower, axis=0)
    with pytest.raises(TypeError, match=over_spec_msg):
        df.rename(str.lower, index=str.lower, axis='columns')
    over_spec_msg = "Cannot specify both 'mapper' and any of 'index' or 'columns'"
    with pytest.raises(TypeError, match=over_spec_msg):
        df.rename(str.lower, index=str.lower, columns=str.lower)
    with pytest.raises(TypeError, match='multiple values'):
        df.rename(id, mapper=id)