import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('fit_intercept', [False, True])
def test_multinomial_identifiability_on_iris(fit_intercept):
    """Test that the multinomial classification is identifiable.

    A multinomial with c classes can be modeled with
    probability_k = exp(X@coef_k) / sum(exp(X@coef_l), l=1..c) for k=1..c.
    This is not identifiable, unless one chooses a further constraint.
    According to [1], the maximum of the L2 penalized likelihood automatically
    satisfies the symmetric constraint:
    sum(coef_k, k=1..c) = 0

    Further details can be found in [2].

    Reference
    ---------
    .. [1] :doi:`Zhu, Ji and Trevor J. Hastie. "Classification of gene microarrays by
           penalized logistic regression". Biostatistics 5 3 (2004): 427-43.
           <10.1093/biostatistics/kxg046>`

    .. [2] :arxiv:`Noah Simon and Jerome Friedman and Trevor Hastie. (2013)
           "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
           Multinomial Regression". <1311.6529>`
    """
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]
    clf = LogisticRegression(C=len(iris.data), solver='lbfgs', multi_class='multinomial', fit_intercept=fit_intercept)
    X_scaled = scale(iris.data)
    clf.fit(X_scaled, target)
    assert_allclose(clf.coef_.sum(axis=0), 0, atol=1e-10)
    if fit_intercept:
        clf.intercept_.sum(axis=0) == pytest.approx(0, abs=1e-15)