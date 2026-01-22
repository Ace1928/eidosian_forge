import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@not_implemented
def test_merge_raises(self):
    a = pd.DataFrame({'A': [0, 1, 2]}, index=['a', 'b', 'c']).set_flags(allows_duplicate_labels=False)
    b = pd.DataFrame({'B': [0, 1, 2]}, index=['a', 'b', 'b'])
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.merge(a, b, left_index=True, right_index=True)