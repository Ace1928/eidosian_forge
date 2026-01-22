import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
Test truncated SVD transformer.