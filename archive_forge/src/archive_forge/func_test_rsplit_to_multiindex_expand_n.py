from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_rsplit_to_multiindex_expand_n():
    idx = Index(['some_equal_splits', 'with_no_nans'])
    result = idx.str.rsplit('_', expand=True, n=1)
    exp = MultiIndex.from_tuples([('some_equal', 'splits'), ('with_no', 'nans')])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 2