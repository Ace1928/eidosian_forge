from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_nosort(self):
    d = {'var1': np.random.default_rng(2).integers(0, 10, size=10), 'var2': np.random.default_rng(2).integers(0, 10, size=10), 'var3': [datetime(2012, 1, 12), datetime(2011, 2, 4), datetime(2010, 2, 3), datetime(2012, 1, 12), datetime(2011, 2, 4), datetime(2012, 4, 3), datetime(2012, 3, 4), datetime(2008, 5, 1), datetime(2010, 2, 3), datetime(2012, 2, 3)]}
    df = DataFrame.from_dict(d)
    var3 = df.var3.unique()
    var3 = np.sort(var3)
    new = DataFrame.from_dict({'var3': var3, 'var8': np.random.default_rng(2).random(7)})
    result = df.merge(new, on='var3', sort=False)
    exp = merge(df, new, on='var3', sort=False)
    tm.assert_frame_equal(result, exp)
    assert (df.var3.unique() == result.var3.unique()).all()