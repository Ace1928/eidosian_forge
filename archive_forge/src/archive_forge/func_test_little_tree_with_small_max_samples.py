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
@pytest.mark.parametrize('ForestClass', [RandomForestClassifier, RandomForestRegressor])
def test_little_tree_with_small_max_samples(ForestClass):
    rng = np.random.RandomState(1)
    X = rng.randn(10000, 2)
    y = rng.randn(10000) > 0
    est1 = ForestClass(n_estimators=1, random_state=rng, max_samples=None)
    est2 = ForestClass(n_estimators=1, random_state=rng, max_samples=2)
    est1.fit(X, y)
    est2.fit(X, y)
    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_
    msg = 'Tree without `max_samples` restriction should have more nodes'
    assert tree1.node_count > tree2.node_count, msg