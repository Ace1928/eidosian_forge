import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dataframe_clip(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
    for lb, ub in [(-1, 1), (1, -1)]:
        clipped_df = df.clip(lb, ub)
        lb, ub = (min(lb, ub), max(ub, lb))
        lb_mask = df.values <= lb
        ub_mask = df.values >= ub
        mask = ~lb_mask & ~ub_mask
        assert (clipped_df.values[lb_mask] == lb).all()
        assert (clipped_df.values[ub_mask] == ub).all()
        assert (clipped_df.values[mask] == df.values[mask]).all()