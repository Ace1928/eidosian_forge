import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy', ['uniform', 'kmeans'])
def test_kbd_subsample_warning(strategy):
    X = np.random.RandomState(0).random_sample((100, 1))
    kbd = KBinsDiscretizer(strategy=strategy, random_state=0)
    with pytest.warns(FutureWarning, match='subsample=200_000 will be used by default'):
        kbd.fit(X)