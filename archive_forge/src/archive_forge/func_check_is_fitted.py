import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to
    specify if the estimator is fitted or not. See
    :ref:`sphx_glr_auto_examples_developing_estimators_sklearn_is_fitted.py`
    for an example on how to use the API.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.utils.validation import check_is_fitted
    >>> from sklearn.exceptions import NotFittedError
    >>> lr = LogisticRegression()
    >>> try:
    ...     check_is_fitted(lr)
    ... except NotFittedError as exc:
    ...     print(f"Model is not fitted yet.")
    Model is not fitted yet.
    >>> lr.fit([[1, 2], [1, 3]], [1, 0])
    LogisticRegression()
    >>> check_is_fitted(lr)
    """
    if isclass(estimator):
        raise TypeError('{} is a class, not an instance.'.format(estimator))
    if msg is None:
        msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    if not hasattr(estimator, 'fit'):
        raise TypeError('%s is not an estimator instance.' % estimator)
    if not _is_fitted(estimator, attributes, all_or_any):
        raise NotFittedError(msg % {'name': type(estimator).__name__})