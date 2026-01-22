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
def test_pickle_version_no_warning_is_issued_with_non_sklearn_estimator():
    iris = datasets.load_iris()
    tree = TreeNoVersion().fit(iris.data, iris.target)
    tree_pickle_noversion = pickle.dumps(tree)
    try:
        module_backup = TreeNoVersion.__module__
        TreeNoVersion.__module__ = 'notsklearn'
        assert_no_warnings(pickle.loads, tree_pickle_noversion)
    finally:
        TreeNoVersion.__module__ = module_backup