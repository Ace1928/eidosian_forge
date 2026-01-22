from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kind', ['series', 'frame'])
@pytest.mark.parametrize('col', ['ints', 'uints', 'labels', 'ts', 'floats'])
def test_at_set_ints_other(self, kind, col, request):
    f = request.getfixturevalue(f'{kind}_{col}')
    indices = generate_indices(f, False)
    for i in indices:
        f.at[i] = 1
        expected = f.loc[i]
        tm.assert_almost_equal(expected, 1)