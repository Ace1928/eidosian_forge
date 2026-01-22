from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, _print_elapsed_time, check_random_state
from ..utils._tags import _safe_tags
from ..utils.metaestimators import _BaseComposition
@property
def named_estimators(self):
    """Dictionary to access any fitted sub-estimators by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
    return Bunch(**dict(self.estimators))