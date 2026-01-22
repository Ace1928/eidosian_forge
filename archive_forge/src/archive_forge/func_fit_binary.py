import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
def fit_binary(est, i, X, y, alpha, C, learning_rate, max_iter, pos_weight, neg_weight, sample_weight, validation_mask=None, random_state=None):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.

    Parameters
    ----------
    est : Estimator object
        The estimator to fit

    i : int
        Index of the positive class

    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, ]
        Target values

    alpha : float
        The regularization parameter

    C : float
        Maximum step size for passive aggressive

    learning_rate : str
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.

    max_iter : int
        The maximum number of iterations (epochs)

    pos_weight : float
        The weight of the positive class

    neg_weight : float
        The weight of the negative class

    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample

    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.

    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    y_i, coef, intercept, average_coef, average_intercept = _prepare_fit_binary(est, y, i, input_dtye=X.dtype)
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]
    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset(X, y_i, sample_weight, random_state=random_state)
    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)
    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i, sample_mask=sample_weight > 0)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    validation_score_cb = est._make_validation_score_cb(validation_mask, X, y_i, sample_weight, classes=classes)
    seed = random_state.randint(MAX_INT)
    tol = est.tol if est.tol is not None else -np.inf
    _plain_sgd = _get_plain_sgd_function(input_dtype=coef.dtype)
    coef, intercept, average_coef, average_intercept, n_iter_ = _plain_sgd(coef, intercept, average_coef, average_intercept, est._loss_function_, penalty_type, alpha, C, est.l1_ratio, dataset, validation_mask, est.early_stopping, validation_score_cb, int(est.n_iter_no_change), max_iter, tol, int(est.fit_intercept), int(est.verbose), int(est.shuffle), seed, pos_weight, neg_weight, learning_rate_type, est.eta0, est.power_t, 0, est.t_, intercept_decay, est.average)
    if est.average:
        if len(est.classes_) == 2:
            est._average_intercept[0] = average_intercept
        else:
            est._average_intercept[i] = average_intercept
    return (coef, intercept, n_iter_)