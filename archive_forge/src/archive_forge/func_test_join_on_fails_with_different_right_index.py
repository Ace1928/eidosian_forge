import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_on_fails_with_different_right_index(self):
    df = DataFrame({'a': np.random.default_rng(2).choice(['m', 'f'], size=3), 'b': np.random.default_rng(2).standard_normal(3)})
    df2 = DataFrame({'a': np.random.default_rng(2).choice(['m', 'f'], size=10), 'b': np.random.default_rng(2).standard_normal(10)}, index=MultiIndex.from_product([range(5), ['A', 'B']]))
    msg = 'len\\(left_on\\) must equal the number of levels in the index of "right"'
    with pytest.raises(ValueError, match=msg):
        merge(df, df2, left_on='a', right_index=True)