from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.experimental.dnn_logit_fn_builder'])
def dnn_logit_fn_builder(units, hidden_units, feature_columns, activation_fn, dropout, input_layer_partitioner, batch_norm):
    """Function builder for a dnn logit_fn.

  Args:
    units: An int indicating the dimension of the logit layer.  In the MultiHead
      case, this should be the sum of all component Heads' logit dimensions.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Partitioner for input layer.
    batch_norm: Whether to use batch normalization after each hidden layer.

  Returns:
    A logit_fn (see below).

  Raises:
    ValueError: If units is not an int.
  """
    if not isinstance(units, six.integer_types):
        raise ValueError('units must be an int.  Given type: {}'.format(type(units)))

    def dnn_logit_fn(features, mode):
        """Deep Neural Network logit_fn.

    Args:
      features: This is the first item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
        `dict` of same.
      mode: Optional. Specifies if this training, evaluation or prediction. See
        `ModeKeys`.

    Returns:
      A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    """
        dnn_model = _DNNModel(units, hidden_units, feature_columns, activation_fn, dropout, input_layer_partitioner, batch_norm, name='dnn')
        return dnn_model(features, mode)
    return dnn_logit_fn