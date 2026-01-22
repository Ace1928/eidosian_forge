from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
def print_tree_recurse(node, depth):
    indent = ('|' + ' ' * spacing) * depth
    indent = indent[:-spacing] + '-' * spacing
    value = None
    if tree_.n_outputs == 1:
        value = tree_.value[node][0]
    else:
        value = tree_.value[node].T[0]
    class_name = np.argmax(value)
    if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
        class_name = class_names[class_name]
    weighted_n_node_samples = tree_.weighted_n_node_samples[node]
    if depth <= max_depth + 1:
        info_fmt = ''
        info_fmt_left = info_fmt
        info_fmt_right = info_fmt
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names_[node]
            threshold = tree_.threshold[node]
            threshold = '{1:.{0}f}'.format(decimals, threshold)
            export_text.report += right_child_fmt.format(indent, name, threshold)
            export_text.report += info_fmt_left
            print_tree_recurse(tree_.children_left[node], depth + 1)
            export_text.report += left_child_fmt.format(indent, name, threshold)
            export_text.report += info_fmt_right
            print_tree_recurse(tree_.children_right[node], depth + 1)
        else:
            _add_leaf(value, weighted_n_node_samples, class_name, indent)
    else:
        subtree_depth = _compute_depth(tree_, node)
        if subtree_depth == 1:
            _add_leaf(value, weighted_n_node_samples, class_name, indent)
        else:
            trunc_report = 'truncated branch of depth %d' % subtree_depth
            export_text.report += truncation_fmt.format(indent, trunc_report)