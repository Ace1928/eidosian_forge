from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.confusion_matrix', v1=[])
@dispatch.add_dispatch_support
def confusion_matrix(labels, predictions, num_classes=None, weights=None, dtype=dtypes.int32, name=None):
    """Computes the confusion matrix from predictions and labels.

  The matrix columns represent the prediction labels and the rows represent the
  real labels. The confusion matrix is always a 2-D array of shape `[n, n]`,
  where `n` is the number of valid labels for a given classification task. Both
  prediction and labels must be 1-D arrays of the same shape in order for this
  function to work.

  If `num_classes` is `None`, then `num_classes` will be set to one plus the
  maximum value in either predictions or labels. Class labels are expected to
  start at 0. For example, if `num_classes` is 3, then the possible labels
  would be `[0, 1, 2]`.

  If `weights` is not `None`, then each prediction contributes its
  corresponding weight to the total value of the confusion matrix cell.

  For example:

  ```python
    tf.math.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
        [[0 0 0 0 0]
         [0 0 1 0 0]
         [0 0 1 0 0]
         [0 0 0 0 0]
         [0 0 0 0 1]]
  ```

  Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
  resulting in a 5x5 confusion matrix.

  Args:
    labels: 1-D `Tensor` of real labels for the classification task.
    predictions: 1-D `Tensor` of predictions for a given classification.
    num_classes: The possible number of labels the classification task can
                 have. If this value is not provided, it will be calculated
                 using both predictions and labels array.
    weights: An optional `Tensor` whose shape matches `predictions`.
    dtype: Data type of the confusion matrix.
    name: Scope name.

  Returns:
    A `Tensor` of type `dtype` with shape `[n, n]` representing the confusion
    matrix, where `n` is the number of possible labels in the classification
    task.

  Raises:
    ValueError: If both predictions and labels are not 1-D vectors and have
      mismatched shapes, or if `weights` is not `None` and its shape doesn't
      match `predictions`.
  """
    with ops.name_scope(name, 'confusion_matrix', (predictions, labels, num_classes, weights)) as name:
        labels, predictions = remove_squeezable_dimensions(ops.convert_to_tensor(labels, name='labels'), ops.convert_to_tensor(predictions, name='predictions'))
        predictions = math_ops.cast(predictions, dtypes.int64)
        labels = math_ops.cast(labels, dtypes.int64)
        labels = control_flow_ops.with_dependencies([check_ops.assert_non_negative(labels, message='`labels` contains negative values')], labels)
        predictions = control_flow_ops.with_dependencies([check_ops.assert_non_negative(predictions, message='`predictions` contains negative values')], predictions)
        if num_classes is None:
            num_classes = math_ops.maximum(math_ops.reduce_max(predictions), math_ops.reduce_max(labels)) + 1
        else:
            num_classes_int64 = math_ops.cast(num_classes, dtypes.int64)
            labels = control_flow_ops.with_dependencies([check_ops.assert_less(labels, num_classes_int64, message='`labels` out of bound')], labels)
            predictions = control_flow_ops.with_dependencies([check_ops.assert_less(predictions, num_classes_int64, message='`predictions` out of bound')], predictions)
        if weights is not None:
            weights = ops.convert_to_tensor(weights, name='weights')
            predictions.get_shape().assert_is_compatible_with(weights.get_shape())
            weights = math_ops.cast(weights, dtype)
        shape = array_ops_stack.stack([num_classes, num_classes])
        indices = array_ops_stack.stack([labels, predictions], axis=1)
        values = array_ops.ones_like(predictions, dtype) if weights is None else weights
        return array_ops.scatter_nd(indices=indices, updates=values, shape=math_ops.cast(shape, dtypes.int64))