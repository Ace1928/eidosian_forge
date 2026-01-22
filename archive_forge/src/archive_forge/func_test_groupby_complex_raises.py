import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func', ['min', 'max', 'var'])
def test_groupby_complex_raises(func):
    data = Series(np.arange(20).reshape(10, 2).dot([1, 2j]))
    msg = 'No matching signature found'
    with pytest.raises(TypeError, match=msg):
        data.groupby(data.index % 2).agg(func)