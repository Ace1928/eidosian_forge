import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_estimator_getstate_using_slots_error_message():
    """Using a `BaseEstimator` with `__slots__` is not supported."""

    class WithSlots:
        __slots__ = ('x',)

    class Estimator(BaseEstimator, WithSlots):
        pass
    msg = 'You cannot use `__slots__` in objects inheriting from `sklearn.base.BaseEstimator`'
    with pytest.raises(TypeError, match=msg):
        Estimator().__getstate__()
    with pytest.raises(TypeError, match=msg):
        pickle.dumps(Estimator())