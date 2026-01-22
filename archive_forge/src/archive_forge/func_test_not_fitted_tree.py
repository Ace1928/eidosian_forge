from io import StringIO
from re import finditer, search
from textwrap import dedent
import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
def test_not_fitted_tree(pyplot):
    clf = DecisionTreeRegressor()
    with pytest.raises(NotFittedError):
        plot_tree(clf)