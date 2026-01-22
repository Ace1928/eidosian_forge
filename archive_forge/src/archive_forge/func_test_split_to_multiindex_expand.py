from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_to_multiindex_expand():
    idx = Index(['some_equal_splits', 'with_no_nans', np.nan, None])
    result = idx.str.split('_', expand=True)
    exp = MultiIndex.from_tuples([('some', 'equal', 'splits'), ('with', 'no', 'nans'), [np.nan, np.nan, np.nan], [None, None, None]])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3