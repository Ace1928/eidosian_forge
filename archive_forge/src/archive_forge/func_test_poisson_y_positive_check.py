import itertools
import math
import pickle
from collections import defaultdict
from functools import partial
from itertools import combinations, product
from typing import Any, Dict
from unittest.mock import patch
import joblib
import numpy as np
import pytest
from scipy.special import comb
import sklearn
from sklearn import clone, datasets
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._forest import (
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree._classes import SPARSE_SPLITTERS
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel
from sklearn.utils.validation import check_random_state
def test_poisson_y_positive_check():
    est = RandomForestRegressor(criterion='poisson')
    X = np.zeros((3, 3))
    y = [-1, 1, 3]
    err_msg = 'Some value\\(s\\) of y are negative which is not allowed for Poisson regression.'
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)
    y = [0, 0, 0]
    err_msg = 'Sum of y is not strictly positive which is necessary for Poisson regression.'
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)