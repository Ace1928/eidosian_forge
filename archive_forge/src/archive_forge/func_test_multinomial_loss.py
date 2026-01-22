import math
import re
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import make_dataset
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS
def test_multinomial_loss():
    X, y = (iris.data, iris.target.astype(np.float64))
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    rng = check_random_state(42)
    weights = rng.randn(n_features, n_classes)
    intercept = rng.randn(n_classes)
    sample_weights = np.abs(rng.randn(n_samples))
    dataset, _ = make_dataset(X, y, sample_weights, random_state=42)
    loss_1, grad_1 = _multinomial_grad_loss_all_samples(dataset, weights, intercept, n_samples, n_features, n_classes)
    loss = LinearModelLoss(base_loss=HalfMultinomialLoss(n_classes=n_classes), fit_intercept=True)
    weights_intercept = np.vstack((weights, intercept)).T
    loss_2, grad_2 = loss.loss_gradient(weights_intercept, X, y, l2_reg_strength=0.0, sample_weight=sample_weights)
    grad_2 = grad_2[:, :-1].T
    loss_2 *= np.sum(sample_weights)
    grad_2 *= np.sum(sample_weights)
    assert_array_almost_equal(grad_1, grad_2)
    assert_almost_equal(loss_1, loss_2)