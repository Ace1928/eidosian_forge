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
@pytest.mark.parametrize('tree,dataset', [(DecisionTreeClassifier(max_depth=2, random_state=0), datasets.make_classification(random_state=0)), (DecisionTreeRegressor(max_depth=2, random_state=0), datasets.make_regression(random_state=0))])
def test_score_sample_weight(tree, dataset):
    rng = np.random.RandomState(0)
    X, y = dataset
    tree.fit(X, y)
    sample_weight = rng.randint(1, 10, size=len(y))
    score_unweighted = tree.score(X, y)
    score_weighted = tree.score(X, y, sample_weight=sample_weight)
    msg = 'Unweighted and weighted scores are unexpectedly equal'
    assert score_unweighted != score_weighted, msg