from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kind', ['series', 'frame'])
@pytest.mark.parametrize('col', ['labels', 'ts', 'floats'])
def test_iat_set_other(self, kind, col, request):
    f = request.getfixturevalue(f'{kind}_{col}')
    msg = 'iAt based indexing can only have integer indexers'
    with pytest.raises(ValueError, match=msg):
        idx = next(generate_indices(f, False))
        f.iat[idx] = 1