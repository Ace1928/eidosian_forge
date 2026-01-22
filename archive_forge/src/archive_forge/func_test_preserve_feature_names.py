import re
import sys
import warnings
from io import StringIO
import joblib
import numpy as np
import pytest
from numpy.testing import (
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('Estimator', [MLPClassifier, MLPRegressor])
def test_preserve_feature_names(Estimator):
    """Check that feature names are preserved when early stopping is enabled.

    Feature names are required for consistency checks during scoring.

    Non-regression test for gh-24846
    """
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    X = pd.DataFrame(data=rng.randn(10, 2), columns=['colname_a', 'colname_b'])
    y = pd.Series(data=np.full(10, 1), name='colname_y')
    model = Estimator(early_stopping=True, validation_fraction=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        model.fit(X, y)