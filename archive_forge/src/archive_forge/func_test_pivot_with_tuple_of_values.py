from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.parametrize('method', [True, False])
def test_pivot_with_tuple_of_values(self, method):
    df = DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    with pytest.raises(KeyError, match="^\\('bar', 'baz'\\)$"):
        if method:
            df.pivot(index='zoo', columns='foo', values=('bar', 'baz'))
        else:
            pd.pivot(df, index='zoo', columns='foo', values=('bar', 'baz'))