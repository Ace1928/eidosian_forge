import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mixed_type_join_with_suffix(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 6)), columns=['a', 'b', 'c', 'd', 'e', 'f'])
    df.insert(0, 'id', 0)
    df.insert(5, 'dt', 'foo')
    grouped = df.groupby('id')
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        grouped.mean()
    mn = grouped.mean(numeric_only=True)
    cn = grouped.count()
    mn.join(cn, rsuffix='_right')