from functools import partial
import numpy as np
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
Test the rcv1 loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs).