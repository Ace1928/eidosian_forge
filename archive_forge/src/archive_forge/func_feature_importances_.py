import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
@property
def feature_importances_(self):
    """The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
    if self.estimators_ is None or len(self.estimators_) == 0:
        raise ValueError('Estimator not fitted, call `fit` before `feature_importances_`.')
    try:
        norm = self.estimator_weights_.sum()
        return sum((weight * clf.feature_importances_ for weight, clf in zip(self.estimator_weights_, self.estimators_))) / norm
    except AttributeError as e:
        raise AttributeError('Unable to compute feature importances since estimator does not have a feature_importances_ attribute') from e