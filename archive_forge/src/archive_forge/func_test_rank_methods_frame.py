from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ax', [0, 1])
@pytest.mark.parametrize('m', ['average', 'min', 'max', 'first', 'dense'])
def test_rank_methods_frame(self, ax, m):
    sp_stats = pytest.importorskip('scipy.stats')
    xs = np.random.default_rng(2).integers(0, 21, (100, 26))
    xs = (xs - 10.0) / 10.0
    cols = [chr(ord('z') - i) for i in range(xs.shape[1])]
    for vals in [xs, xs + 1000000.0, xs * 1e-06]:
        df = DataFrame(vals, columns=cols)
        result = df.rank(axis=ax, method=m)
        sprank = np.apply_along_axis(sp_stats.rankdata, ax, vals, m if m != 'first' else 'ordinal')
        sprank = sprank.astype(np.float64)
        expected = DataFrame(sprank, columns=cols).astype('float64')
        tm.assert_frame_equal(result, expected)