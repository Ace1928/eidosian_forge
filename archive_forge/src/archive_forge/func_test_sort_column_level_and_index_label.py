import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_column_level_and_index_label(self, df_none, df_idx, sort_names, ascending, request):
    levels = df_idx.index.names
    expected = df_none.sort_values(by=sort_names, ascending=ascending, axis=0).set_index(levels).T
    result = df_idx.T.sort_values(by=sort_names, ascending=ascending, axis=1)
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    tm.assert_frame_equal(result, expected)