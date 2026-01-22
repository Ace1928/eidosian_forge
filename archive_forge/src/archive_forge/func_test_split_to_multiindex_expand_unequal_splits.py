from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_to_multiindex_expand_unequal_splits():
    idx = Index(['some_unequal_splits', 'one_of_these_things_is_not', np.nan, None])
    result = idx.str.split('_', expand=True)
    exp = MultiIndex.from_tuples([('some', 'unequal', 'splits', np.nan, np.nan, np.nan), ('one', 'of', 'these', 'things', 'is', 'not'), (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (None, None, None, None, None, None)])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 6
    with pytest.raises(ValueError, match='expand must be'):
        idx.str.split('_', expand='not_a_boolean')