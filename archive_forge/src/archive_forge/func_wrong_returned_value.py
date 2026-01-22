import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
def wrong_returned_value(x, y):
    return '1'