from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('group_dropna, count_dropna, expected_rows, expected_values', [(False, False, [0, 1, 3, 5, 7, 6, 8, 2, 4], [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0]), (False, True, [0, 1, 3, 5, 2, 4], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]), (True, False, [0, 1, 5, 7, 6, 8], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]), (True, True, [0, 1, 5], [0.5, 0.5, 1.0])])
def test_dropna_combinations(nulls_df, group_dropna, count_dropna, expected_rows, expected_values, request):
    if Version(np.__version__) >= Version('1.25') and (not group_dropna):
        request.node.add_marker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    gp = nulls_df.groupby(['A', 'B'], dropna=group_dropna)
    result = gp.value_counts(normalize=True, sort=True, dropna=count_dropna)
    columns = DataFrame()
    for column in nulls_df.columns:
        columns[column] = [nulls_df[column][row] for row in expected_rows]
    index = MultiIndex.from_frame(columns)
    expected = Series(data=expected_values, index=index, name='proportion')
    tm.assert_series_equal(result, expected)