from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_class_distribution(csc_container):
    y = np.array([[1, 0, 0, 1], [2, 2, 0, 1], [1, 3, 0, 1], [4, 2, 0, 1], [2, 0, 0, 1], [1, 3, 0, 1]])
    data = np.array([1, 2, 1, 4, 2, 1, 0, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1])
    indices = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 5])
    indptr = np.array([0, 6, 11, 11, 17])
    y_sp = csc_container((data, indices, indptr), shape=(6, 4))
    classes, n_classes, class_prior = class_distribution(y)
    classes_sp, n_classes_sp, class_prior_sp = class_distribution(y_sp)
    classes_expected = [[1, 2, 4], [0, 2, 3], [0], [1]]
    n_classes_expected = [3, 3, 1, 1]
    class_prior_expected = [[3 / 6, 2 / 6, 1 / 6], [1 / 3, 1 / 3, 1 / 3], [1.0], [1.0]]
    for k in range(y.shape[1]):
        assert_array_almost_equal(classes[k], classes_expected[k])
        assert_array_almost_equal(n_classes[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior[k], class_prior_expected[k])
        assert_array_almost_equal(classes_sp[k], classes_expected[k])
        assert_array_almost_equal(n_classes_sp[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior_sp[k], class_prior_expected[k])
    classes, n_classes, class_prior = class_distribution(y, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    classes_sp, n_classes_sp, class_prior_sp = class_distribution(y, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    class_prior_expected = [[4 / 9, 3 / 9, 2 / 9], [2 / 9, 4 / 9, 3 / 9], [1.0], [1.0]]
    for k in range(y.shape[1]):
        assert_array_almost_equal(classes[k], classes_expected[k])
        assert_array_almost_equal(n_classes[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior[k], class_prior_expected[k])
        assert_array_almost_equal(classes_sp[k], classes_expected[k])
        assert_array_almost_equal(n_classes_sp[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior_sp[k], class_prior_expected[k])