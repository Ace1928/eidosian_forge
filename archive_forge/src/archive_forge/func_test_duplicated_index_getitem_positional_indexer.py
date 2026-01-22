from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('index_vals', ['aabcd', 'aadcb'])
def test_duplicated_index_getitem_positional_indexer(index_vals):
    s = Series(range(5), index=list(index_vals))
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s[3]
    assert result == 3