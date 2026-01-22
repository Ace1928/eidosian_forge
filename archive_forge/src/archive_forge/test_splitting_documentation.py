import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
Check that feature_fraction_per_split is respected.

    Because we set `n_features = 4` and `feature_fraction_per_split = 0.25`, it means
    that calling `splitter.find_node_split` will be allowed to select a split for a
    single completely random feature at each call. So if we iterate enough, we should
    cover all the allowed features, irrespective of the values of the gradients and
    Hessians of the objective.
    