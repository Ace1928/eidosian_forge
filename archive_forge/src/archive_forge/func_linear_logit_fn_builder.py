from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.experimental.linear_logit_fn_builder'])
def linear_logit_fn_builder(units, feature_columns, sparse_combiner='sum'):
    """Function builder for a linear logit_fn.

  Args:
    units: An int indicating the dimension of the logit layer.
    feature_columns: An iterable containing all the feature columns used by the
      model.
    sparse_combiner: A string specifying how to reduce if a categorical column
      is multivalent.  One of "mean", "sqrtn", and "sum".

  Returns:
    A logit_fn (see below).

  """

    def linear_logit_fn(features):
        """Linear model logit_fn.

    Args:
      features: This is the first item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
        `dict` of same.

    Returns:
      A `Tensor` representing the logits.
    """
        if feature_column_lib.is_feature_column_v2(feature_columns):
            linear_model = LinearModel(feature_columns=feature_columns, units=units, sparse_combiner=sparse_combiner, name='linear_model')
            logits = linear_model(features)
            bias = linear_model.bias
            variables = linear_model.variables
            bias = _get_expanded_variable_list([bias])
            variables = _get_expanded_variable_list(variables)
            variables = [var for var in variables if var not in bias]
            bias = _get_expanded_variable_list([bias])
        else:
            linear_model = feature_column._LinearModel(feature_columns=feature_columns, units=units, sparse_combiner=sparse_combiner, name='linear_model')
            logits = linear_model(features)
            cols_to_vars = linear_model.cols_to_vars()
            bias = cols_to_vars.pop('bias')
            variables = cols_to_vars.values()
            variables = _get_expanded_variable_list(variables)
        if units > 1:
            tf.compat.v1.summary.histogram('bias', bias)
        else:
            tf.compat.v1.summary.scalar('bias', bias[0][0])
        tf.compat.v1.summary.scalar('fraction_of_zero_weights', _compute_fraction_of_zero(variables))
        return logits
    return linear_logit_fn