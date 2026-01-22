import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def gradient_hessian(self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1, gradient_out=None, hessian_out=None, raw_prediction=None):
    """Computes gradient and hessian w.r.t. coef.

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
        gradient_out : None or ndarray of shape coef.shape
            A location into which the gradient is stored. If None, a new array
            might be created.
        hessian_out : None or ndarray
            A location into which the hessian is stored. If None, a new array
            might be created.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of             shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        gradient : ndarray of shape coef.shape
             The gradient of the loss.

        hessian : ndarray
            Hessian matrix.

        hessian_warning : bool
            True if pointwise hessian has more than half of its elements non-positive.
        """
    n_samples, n_features = X.shape
    n_dof = n_features + int(self.fit_intercept)
    if raw_prediction is None:
        weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
    else:
        weights, intercept = self.weight_intercept(coef)
    grad_pointwise, hess_pointwise = self.base_loss.gradient_hessian(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=n_threads)
    sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
    grad_pointwise /= sw_sum
    hess_pointwise /= sw_sum
    hessian_warning = np.mean(hess_pointwise <= 0) > 0.25
    hess_pointwise = np.abs(hess_pointwise)
    if not self.base_loss.is_multiclass:
        if gradient_out is None:
            grad = np.empty_like(coef, dtype=weights.dtype)
        else:
            grad = gradient_out
        grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
        if self.fit_intercept:
            grad[-1] = grad_pointwise.sum()
        if hessian_out is None:
            hess = np.empty(shape=(n_dof, n_dof), dtype=weights.dtype)
        else:
            hess = hessian_out
        if hessian_warning:
            return (grad, hess, hessian_warning)
        if sparse.issparse(X):
            hess[:n_features, :n_features] = (X.T @ sparse.dia_matrix((hess_pointwise, 0), shape=(n_samples, n_samples)) @ X).toarray()
        else:
            WX = hess_pointwise[:, None] * X
            hess[:n_features, :n_features] = np.dot(X.T, WX)
        if l2_reg_strength > 0:
            hess.reshape(-1)[:n_features * n_dof:n_dof + 1] += l2_reg_strength
        if self.fit_intercept:
            Xh = X.T @ hess_pointwise
            hess[:-1, -1] = Xh
            hess[-1, :-1] = Xh
            hess[-1, -1] = hess_pointwise.sum()
    else:
        raise NotImplementedError
    return (grad, hess, hessian_warning)