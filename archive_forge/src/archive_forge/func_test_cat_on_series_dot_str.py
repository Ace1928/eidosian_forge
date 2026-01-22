import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
def test_cat_on_series_dot_str():
    ps = Series(['AbC', 'de', 'FGHI', 'j', 'kLLLm'])
    message = re.escape('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')
    with pytest.raises(TypeError, match=message):
        ps.str.cat(others=ps.str)