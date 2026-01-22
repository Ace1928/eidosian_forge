import re
import numpy as np
import pytest
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
def assert_children_values_bounded(grower, monotonic_cst):
    if monotonic_cst == MonotonicConstraint.NO_CST:
        return

    def recursively_check_children_node_values(node, right_sibling=None):
        if node.is_leaf:
            return
        if right_sibling is not None:
            middle = (node.value + right_sibling.value) / 2
            if monotonic_cst == MonotonicConstraint.POS:
                assert node.left_child.value <= node.right_child.value <= middle
                if not right_sibling.is_leaf:
                    assert middle <= right_sibling.left_child.value <= right_sibling.right_child.value
            else:
                assert node.left_child.value >= node.right_child.value >= middle
                if not right_sibling.is_leaf:
                    assert middle >= right_sibling.left_child.value >= right_sibling.right_child.value
        recursively_check_children_node_values(node.left_child, right_sibling=node.right_child)
        recursively_check_children_node_values(node.right_child)
    recursively_check_children_node_values(grower.root)