import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_method(self):
    idx1 = CategoricalIndex(list('aabcde'), categories=list('edabc'))
    idx2 = CategoricalIndex(list('abf'))
    msg = 'method pad not yet implemented for CategoricalIndex'
    with pytest.raises(NotImplementedError, match=msg):
        idx2.get_indexer(idx1, method='pad')
    msg = 'method backfill not yet implemented for CategoricalIndex'
    with pytest.raises(NotImplementedError, match=msg):
        idx2.get_indexer(idx1, method='backfill')
    msg = 'method nearest not yet implemented for CategoricalIndex'
    with pytest.raises(NotImplementedError, match=msg):
        idx2.get_indexer(idx1, method='nearest')