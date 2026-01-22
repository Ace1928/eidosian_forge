import numpy as np
import pytest
from scipy import sparse
from sklearn import base, datasets, linear_model, svm
from sklearn.datasets import load_digits, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm.tests import test_svm
from sklearn.utils._testing import (
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import (
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_realdata(csr_container):
    data = np.array([0.03771744, 0.1003567, 0.01174647, 0.027069])
    indices = np.array([6, 5, 35, 31], dtype=np.int32)
    indptr = np.array([0] * 8 + [1] * 32 + [2] * 38 + [4] * 3, dtype=np.int32)
    X = csr_container((data, indices, indptr))
    y = np.array([1.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 3.0, 1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 0.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0])
    clf = svm.SVC(kernel='linear').fit(X.toarray(), y)
    sp_clf = svm.SVC(kernel='linear').fit(X.tocoo(), y)
    assert_array_equal(clf.support_vectors_, sp_clf.support_vectors_.toarray())
    assert_array_equal(clf.dual_coef_, sp_clf.dual_coef_.toarray())