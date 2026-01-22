import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_dataframe_different_dtypes(self, using_infer_string):
    df1 = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
    df2 = DataFrame({'a': [1, 2, 3]})
    result = concat([df1[:0], df2[:0]])
    assert result['a'].dtype == np.int64
    assert result['b'].dtype == np.object_ if not using_infer_string else 'string'