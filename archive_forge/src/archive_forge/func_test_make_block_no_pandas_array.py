from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
def test_make_block_no_pandas_array(block_maker):
    arr = pd.arrays.NumpyExtensionArray(np.array([1, 2]))
    result = block_maker(arr, BlockPlacement(slice(len(arr))), ndim=arr.ndim)
    assert result.dtype.kind in ['i', 'u']
    if block_maker is make_block:
        assert result.is_extension is False
        result = block_maker(arr, slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim)
        assert result.dtype.kind in ['i', 'u']
        assert result.is_extension is False
        result = block_maker(arr.to_numpy(), slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim)
        assert result.dtype.kind in ['i', 'u']
        assert result.is_extension is False