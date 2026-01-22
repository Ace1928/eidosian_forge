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
def test_merge_common(self, df, df2):
    joined = merge(df, df2)
    exp = merge(df, df2, on=['key1', 'key2'])
    tm.assert_frame_equal(joined, exp)