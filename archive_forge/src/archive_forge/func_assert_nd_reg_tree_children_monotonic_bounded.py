import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
def assert_nd_reg_tree_children_monotonic_bounded(tree_, monotonic_cst):
    upper_bound = np.full(tree_.node_count, np.inf)
    lower_bound = np.full(tree_.node_count, -np.inf)
    for i in range(tree_.node_count):
        feature = tree_.feature[i]
        node_value = tree_.value[i][0][0]
        assert np.float32(node_value) <= np.float32(upper_bound[i])
        assert np.float32(node_value) >= np.float32(lower_bound[i])
        if feature < 0:
            continue
        i_left = tree_.children_left[i]
        i_right = tree_.children_right[i]
        middle_value = (tree_.value[i_left][0][0] + tree_.value[i_right][0][0]) / 2
        if monotonic_cst[feature] == 0:
            lower_bound[i_left] = lower_bound[i]
            upper_bound[i_left] = upper_bound[i]
            lower_bound[i_right] = lower_bound[i]
            upper_bound[i_right] = upper_bound[i]
        elif monotonic_cst[feature] == 1:
            assert tree_.value[i_left] <= tree_.value[i_right]
            lower_bound[i_left] = lower_bound[i]
            upper_bound[i_left] = middle_value
            lower_bound[i_right] = middle_value
            upper_bound[i_right] = upper_bound[i]
        elif monotonic_cst[feature] == -1:
            assert tree_.value[i_left] >= tree_.value[i_right]
            lower_bound[i_left] = middle_value
            upper_bound[i_left] = upper_bound[i]
            lower_bound[i_right] = lower_bound[i]
            upper_bound[i_right] = middle_value
        else:
            raise ValueError(f'monotonic_cst[{feature}]={monotonic_cst[feature]}')