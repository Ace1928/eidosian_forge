from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
def compute_depth_(current_node, current_depth, children_left, children_right, depths):
    depths += [current_depth]
    left = children_left[current_node]
    right = children_right[current_node]
    if left != -1 and right != -1:
        compute_depth_(left, current_depth + 1, children_left, children_right, depths)
        compute_depth_(right, current_depth + 1, children_left, children_right, depths)