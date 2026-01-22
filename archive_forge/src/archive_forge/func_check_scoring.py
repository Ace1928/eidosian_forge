import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
@validate_params({'estimator': [HasMethods('fit')], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'allow_none': ['boolean']}, prefer_skip_nested_validation=True)
def check_scoring(estimator, scoring=None, *, allow_none=False):
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.

    allow_none : bool, default=False
        If no scoring is specified and the estimator has no score function, we
        can either return None or raise an exception.

    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.metrics import check_scoring
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> classifier = DecisionTreeClassifier(max_depth=2).fit(X, y)
    >>> scorer = check_scoring(classifier, scoring='accuracy')
    >>> scorer(classifier, X, y)
    0.96...
    """
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if callable(scoring):
        module = getattr(scoring, '__module__', None)
        if hasattr(module, 'startswith') and module.startswith('sklearn.metrics.') and (not module.startswith('sklearn.metrics._scorer')) and (not module.startswith('sklearn.metrics.tests.')):
            raise ValueError('scoring value %r looks like it is a metric function rather than a scorer. A scorer should require an estimator as its first parameter. Please use `make_scorer` to convert a metric to a scorer.' % scoring)
        return get_scorer(scoring)
    if scoring is None:
        if hasattr(estimator, 'score'):
            return _PassthroughScorer(estimator)
        elif allow_none:
            return None
        else:
            raise TypeError("If no scoring is specified, the estimator passed should have a 'score' method. The estimator %r does not." % estimator)