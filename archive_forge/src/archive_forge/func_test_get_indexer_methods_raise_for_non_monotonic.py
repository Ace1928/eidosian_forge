from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['pad', 'ffill', 'backfill', 'bfill', 'nearest'])
def test_get_indexer_methods_raise_for_non_monotonic(self, method):
    mi = MultiIndex.from_arrays([[0, 4, 2], [0, 4, 2]])
    if method == 'nearest':
        err = NotImplementedError
        msg = 'not implemented yet for MultiIndex'
    else:
        err = ValueError
        msg = 'index must be monotonic increasing or decreasing'
    with pytest.raises(err, match=msg):
        mi.get_indexer([(1, 1)], method=method)