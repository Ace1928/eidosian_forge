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
def sag(X, y, step_size, alpha, n_iter=1, dloss=None, sparse=False, sample_weight=None, fit_intercept=True, saga=False):
    n_samples, n_features = (X.shape[0], X.shape[1])
    weights = np.zeros(X.shape[1])
    sum_gradient = np.zeros(X.shape[1])
    gradient_memory = np.zeros((n_samples, n_features))
    intercept = 0.0
    intercept_sum_gradient = 0.0
    intercept_gradient_memory = np.zeros(n_samples)
    rng = np.random.RandomState(77)
    decay = 1.0
    seen = set()
    if sparse:
        decay = 0.01
    for epoch in range(n_iter):
        for k in range(n_samples):
            idx = int(rng.rand() * n_samples)
            entry = X[idx]
            seen.add(idx)
            p = np.dot(entry, weights) + intercept
            gradient = dloss(p, y[idx])
            if sample_weight is not None:
                gradient *= sample_weight[idx]
            update = entry * gradient + alpha * weights
            gradient_correction = update - gradient_memory[idx]
            sum_gradient += gradient_correction
            gradient_memory[idx] = update
            if saga:
                weights -= gradient_correction * step_size * (1 - 1.0 / len(seen))
            if fit_intercept:
                gradient_correction = gradient - intercept_gradient_memory[idx]
                intercept_gradient_memory[idx] = gradient
                intercept_sum_gradient += gradient_correction
                gradient_correction *= step_size * (1.0 - 1.0 / len(seen))
                if saga:
                    intercept -= step_size * intercept_sum_gradient / len(seen) * decay + gradient_correction
                else:
                    intercept -= step_size * intercept_sum_gradient / len(seen) * decay
            weights -= step_size * sum_gradient / len(seen)
    return (weights, intercept)