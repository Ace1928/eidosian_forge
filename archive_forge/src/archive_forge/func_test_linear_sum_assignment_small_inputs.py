from numpy.testing import assert_array_equal
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_linear_sum_assignment_small_inputs(sign, test_case):
    linear_sum_assignment_assertions(linear_sum_assignment, np.array, sign, test_case)