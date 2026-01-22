from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.missing import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:elementwise comp:DeprecationWarning')
def test_get_indexer_non_unique_np_nats(self, np_nat_fixture, np_nat_fixture2):
    expected_missing = np.array([], dtype=np.intp)
    if is_matching_na(np_nat_fixture, np_nat_fixture2):
        index = Index(np.array(['2021-10-02', np_nat_fixture.copy(), np_nat_fixture2.copy()], dtype=object), dtype=object)
        indexer, missing = index.get_indexer_non_unique(Index([np_nat_fixture], dtype=object))
        expected_indexer = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)
    else:
        try:
            np_nat_fixture == np_nat_fixture2
        except (TypeError, OverflowError):
            return
        index = Index(np.array(['2021-10-02', np_nat_fixture, np_nat_fixture2, np_nat_fixture, np_nat_fixture2], dtype=object), dtype=object)
        indexer, missing = index.get_indexer_non_unique(Index([np_nat_fixture], dtype=object))
        expected_indexer = np.array([1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)