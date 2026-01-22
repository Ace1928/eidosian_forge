import time
import numpy as np
import pytest
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing
Small batches are slow, large batches -> large allocations -> also slow.

    https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691
    