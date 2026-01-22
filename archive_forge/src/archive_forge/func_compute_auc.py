from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def compute_auc(tp, fn, tn, fp, name):
    """Computes the roc-auc or pr-auc based on confusion counts."""
    if curve == 'PR':
        if summation_method == 'trapezoidal':
            logging.warning('Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.')
        elif summation_method == 'careful_interpolation':
            return interpolate_pr_auc(tp, fp, fn)
    rec = math_ops.divide(tp + epsilon, tp + fn + epsilon)
    if curve == 'ROC':
        fp_rate = math_ops.divide(fp, fp + tn + epsilon)
        x = fp_rate
        y = rec
    else:
        prec = math_ops.divide(tp + epsilon, tp + fp + epsilon)
        x = rec
        y = prec
    if summation_method in ('trapezoidal', 'careful_interpolation'):
        return math_ops.reduce_sum(math_ops.multiply(x[:num_thresholds - 1] - x[1:], (y[:num_thresholds - 1] + y[1:]) / 2.0), name=name)
    elif summation_method == 'minoring':
        return math_ops.reduce_sum(math_ops.multiply(x[:num_thresholds - 1] - x[1:], math_ops.minimum(y[:num_thresholds - 1], y[1:])), name=name)
    elif summation_method == 'majoring':
        return math_ops.reduce_sum(math_ops.multiply(x[:num_thresholds - 1] - x[1:], math_ops.maximum(y[:num_thresholds - 1], y[1:])), name=name)
    else:
        raise ValueError(f"Invalid summation_method: {summation_method} summation_method should be 'trapezoidal', 'careful_interpolation', 'minoring', or 'majoring'.")