import numpy as np
import pytest
from pandas import (
from pandas.core.strings.accessor import StringMethods
@pytest.mark.parametrize('dtype', [object, 'category'])
def test_api_per_dtype(index_or_series, dtype, any_skipna_inferred_dtype):
    box = index_or_series
    inferred_dtype, values = any_skipna_inferred_dtype
    t = box(values, dtype=dtype)
    types_passing_constructor = ['string', 'unicode', 'empty', 'bytes', 'mixed', 'mixed-integer']
    if inferred_dtype in types_passing_constructor:
        assert isinstance(t.str, StringMethods)
    else:
        msg = 'Can only use .str accessor with string values.*'
        with pytest.raises(AttributeError, match=msg):
            t.str
        assert not hasattr(t, 'str')