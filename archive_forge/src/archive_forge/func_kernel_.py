from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit
from ..base import BaseEstimator, ClassifierMixin, _fit_context, clone
from ..multiclass import OneVsOneClassifier, OneVsRestClassifier
from ..preprocessing import LabelEncoder
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from .kernels import RBF, CompoundKernel, Kernel
from .kernels import ConstantKernel as C
@property
def kernel_(self):
    """Return the kernel of the base estimator."""
    if self.n_classes_ == 2:
        return self.base_estimator_.kernel_
    else:
        return CompoundKernel([estimator.kernel_ for estimator in self.base_estimator_.estimators_])