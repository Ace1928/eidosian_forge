import re
import numpy as np
import pytest
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
def test_input_error():
    X = [[1, 2], [2, 3], [3, 4]]
    y = [0, 1, 2]
    gbdt = HistGradientBoostingRegressor(monotonic_cst=[1, 0, -1])
    with pytest.raises(ValueError, match=re.escape('monotonic_cst has shape (3,) but the input data')):
        gbdt.fit(X, y)
    for monotonic_cst in ([1, 3], [1, -3], [0.3, -0.7]):
        gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
        expected_msg = re.escape('must be an array-like of -1, 0 or 1. Observed values:')
        with pytest.raises(ValueError, match=expected_msg):
            gbdt.fit(X, y)
    gbdt = HistGradientBoostingClassifier(monotonic_cst=[0, 1])
    with pytest.raises(ValueError, match='monotonic constraints are not supported for multiclass classification'):
        gbdt.fit(X, y)