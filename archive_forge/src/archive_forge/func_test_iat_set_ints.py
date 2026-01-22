from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kind', ['series', 'frame'])
@pytest.mark.parametrize('col', ['ints', 'uints'])
def test_iat_set_ints(self, kind, col, request):
    f = request.getfixturevalue(f'{kind}_{col}')
    indices = generate_indices(f, True)
    for i in indices:
        f.iat[i] = 1
        expected = f.values[i]
        tm.assert_almost_equal(expected, 1)