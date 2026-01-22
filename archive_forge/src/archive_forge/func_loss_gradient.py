import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def loss_gradient(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1, raw_prediction=None):
    """Computes the sum of loss and gradient w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        loss : float
            Weighted average of losses per sample, plus penalty.

        gradient : ndarray of shape coef.shape
             The gradient of the loss.
        """
    (n_samples, n_features), n_classes = (X.shape, self.base_loss.n_classes)
    n_dof = n_features + int(self.fit_intercept)
    if raw_prediction is None:
        weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
    else:
        weights, intercept = self.weight_intercept(coef)
    loss, grad_pointwise = self.base_loss.loss_gradient(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
    sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
    loss = loss.sum() / sw_sum
    loss += self.l2_penalty(weights, l2_reg_strength)
    grad_pointwise /= sw_sum
    if not self.base_loss.is_multiclass:
        grad = np.empty_like(coef, dtype=weights.dtype)
        grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
        if self.fit_intercept:
            grad[-1] = grad_pointwise.sum()
    else:
        grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order='F')
        grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
        if self.fit_intercept:
            grad[:, -1] = grad_pointwise.sum(axis=0)
        if coef.ndim == 1:
            grad = grad.ravel(order='F')
    return (loss, grad)