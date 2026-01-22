from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_index_not_found_raises(index_or_series, any_string_dtype):
    obj = index_or_series(['ABCDEFG', 'BCDEFEF', 'DEFGHIJEF', 'EFGHEF'], dtype=any_string_dtype)
    with pytest.raises(ValueError, match='substring not found'):
        obj.str.index('DE')