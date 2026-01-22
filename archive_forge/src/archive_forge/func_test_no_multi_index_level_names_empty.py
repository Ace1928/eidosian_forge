from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_no_multi_index_level_names_empty(all_parsers):
    parser = all_parsers
    midx = MultiIndex.from_tuples([('A', 1, 2), ('A', 1, 2), ('B', 1, 2)])
    expected = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=midx, columns=['x', 'y', 'z'])
    with tm.ensure_clean() as path:
        expected.to_csv(path)
        result = parser.read_csv(path, index_col=[0, 1, 2])
    tm.assert_frame_equal(result, expected)