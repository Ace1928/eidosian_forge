import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _IS_WASM, IS_PYPY
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csc_container, csr_container', product(CSC_CONTAINERS, CSR_CONTAINERS))
def test_tfidf_transformer_sparse(csc_container, csr_container):
    X = sparse.rand(10, 20000, dtype=np.float64, random_state=42)
    X_csc = csc_container(X)
    X_csr = csr_container(X)
    X_trans_csc = TfidfTransformer().fit_transform(X_csc)
    X_trans_csr = TfidfTransformer().fit_transform(X_csr)
    assert_allclose_dense_sparse(X_trans_csc, X_trans_csr)
    assert X_trans_csc.format == X_trans_csr.format