from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_index_str_accessor_multiindex_raises():
    idx = MultiIndex.from_tuples([('a', 'b'), ('a', 'b')])
    assert idx.inferred_type == 'mixed'
    msg = 'Can only use .str accessor with Index, not MultiIndex'
    with pytest.raises(AttributeError, match=msg):
        idx.str