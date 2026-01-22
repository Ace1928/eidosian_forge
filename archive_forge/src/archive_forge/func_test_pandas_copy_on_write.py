import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
def test_pandas_copy_on_write():
    """
    Test target-encoder cython code when y is read-only.

    The numpy array underlying df["y"] is read-only when copy-on-write is enabled.
    Non-regression test for gh-27879.
    """
    pd = pytest.importorskip('pandas', minversion='2.0')
    with pd.option_context('mode.copy_on_write', True):
        df = pd.DataFrame({'x': ['a', 'b', 'b'], 'y': [4.0, 5.0, 6.0]})
        TargetEncoder(target_type='continuous').fit(df[['x']], df['y'])