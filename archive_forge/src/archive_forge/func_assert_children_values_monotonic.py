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
def assert_children_values_monotonic(predictor, monotonic_cst):
    nodes = predictor.nodes
    left_lower = []
    left_greater = []
    for node in nodes:
        if node['is_leaf']:
            continue
        left_idx = node['left']
        right_idx = node['right']
        if nodes[left_idx]['value'] < nodes[right_idx]['value']:
            left_lower.append(node)
        elif nodes[left_idx]['value'] > nodes[right_idx]['value']:
            left_greater.append(node)
    if monotonic_cst == MonotonicConstraint.NO_CST:
        assert left_lower and left_greater
    elif monotonic_cst == MonotonicConstraint.POS:
        assert left_lower and (not left_greater)
    else:
        assert not left_lower and left_greater