import warnings
import numpy as np
import scipy.sparse as sp
from ..base import _fit_context
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Integral, Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted, check_random_state
from ._k_means_common import _inertia_dense, _inertia_sparse
from ._kmeans import (
def iter_leaves(self):
    """Iterate over all the cluster leaves in the tree."""
    if self.left is None:
        yield self
    else:
        yield from self.left.iter_leaves()
        yield from self.right.iter_leaves()