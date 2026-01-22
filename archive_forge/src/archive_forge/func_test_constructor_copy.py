from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('index', ['string'], indirect=True)
def test_constructor_copy(self, index, using_infer_string):
    arr = np.array(index)
    new_index = Index(arr, copy=True, name='name')
    assert isinstance(new_index, Index)
    assert new_index.name == 'name'
    if using_infer_string:
        tm.assert_extension_array_equal(new_index.values, pd.array(arr, dtype='string[pyarrow_numpy]'))
    else:
        tm.assert_numpy_array_equal(arr, new_index.values)
    arr[0] = 'SOMEBIGLONGSTRING'
    assert new_index[0] != 'SOMEBIGLONGSTRING'