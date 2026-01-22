from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('idx, indexer, check_index_type', [[list('abcde'), [3, 2, 1, 0, 5, 4, 8, 7, 6], True], [list('abcd'), [3, 2, 1, 0, 5, 8, 7, 6], True], [list('abc'), [3, 2, 1, 8, 7, 6], True], [list('eca'), [1, 3, 4, 6, 8], True], [list('edc'), [0, 1, 4, 5, 6], True], [list('eadbc'), [3, 0, 2, 1, 4, 5, 8, 7, 6], True], [list('edwq'), [0, 4, 5], True], [list('wq'), [], False]])
def test_reindex_level_verify(self, idx, indexer, check_index_type):
    df = DataFrame({'jim': list('B' * 4 + 'A' * 2 + 'C' * 3), 'joe': list('abcdeabcd')[::-1], 'jolie': [10, 20, 30] * 3, 'joline': np.random.default_rng(2).integers(0, 1000, 9)})
    icol = ['jim', 'joe', 'jolie']
    left = df.set_index(icol).reindex(idx, level='joe')
    right = df.iloc[indexer].set_index(icol)
    tm.assert_frame_equal(left, right, check_index_type=check_index_type)