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
def test_input_error_related_to_feature_names():
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'a': [0, 1, 2], 'b': [0, 1, 2]})
    y = np.array([0, 1, 0])
    monotonic_cst = {'d': 1, 'a': 1, 'c': -1}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape("monotonic_cst contains 2 unexpected feature names: ['c', 'd'].")
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X, y)
    monotonic_cst = {k: 1 for k in 'abcdefghijklmnopqrstuvwxyz'}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape("monotonic_cst contains 24 unexpected feature names: ['c', 'd', 'e', 'f', 'g', '...'].")
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X, y)
    monotonic_cst = {'a': 1}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape('HistGradientBoostingRegressor was not fitted on data with feature names. Pass monotonic_cst as an integer array instead.')
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X.values, y)
    monotonic_cst = {'b': -1, 'a': '+'}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape("monotonic_cst['a'] must be either -1, 0 or 1. Got '+'.")
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X, y)