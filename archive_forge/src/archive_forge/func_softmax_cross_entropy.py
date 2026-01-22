from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['losses.softmax_cross_entropy'])
@dispatch.add_dispatch_support
def softmax_cross_entropy(onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Note that `onehot_labels` and `logits` must have the same shape,
  e.g. `[batch_size, num_classes]`. The shape of `weights` must be
  broadcastable to loss, whose shape is decided by the shape of `logits`.
  In case the shape of `logits` is `[batch_size, num_classes]`, loss is
  a `Tensor` of shape `[batch_size]`.

  Args:
    onehot_labels: One-hot-encoded labels.
    logits: Logits outputs of the network.
    weights: Optional `Tensor` that is broadcastable to loss.
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has shape `[batch_size]`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.  Also if
      `onehot_labels` or `logits` is None.

  @compatibility(TF2)

  `tf.compat.v1.losses.softmax_cross_entropy` is mostly compatible with eager
  execution and `tf.function`. But, the `loss_collection` argument is
  ignored when executing eagerly and no loss will be written to the loss
  collections. You will need to either hold on to the return value manually
  or rely on `tf.keras.Model` loss tracking.


  To switch to native TF2 style, instantiate the
   `tf.keras.losses.CategoricalCrossentropy` class with `from_logits` set
  as `True` and call the object instead.


  #### Structural Mapping to Native TF2

  Before:

  ```python
  loss = tf.compat.v1.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels,
    logits=logits,
    weights=weights,
    label_smoothing=smoothing)
  ```

  After:

  ```python
  loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=smoothing)
  loss = loss_fn(
    y_true=onehot_labels,
    y_pred=logits,
    sample_weight=weights)
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name     | Note                       |
  | :-------------------- | :--------------- | :------------------------- |
  |  -                    | `from_logits`    | Set `from_logits` as True  |
  :                       :                  : to have identical behavior :
  | `onehot_labels`       | `y_true`         | In `__call__()` method     |
  | `logits`              | `y_pred`         | In `__call__()` method     |
  | `weights`             | `sample_weight`  | In `__call__()` method     |
  | `label_smoothing`     | `label_smoothing`| In constructor             |
  | `scope`               | Not supported    | -                          |
  | `loss_collection`     | Not supported    | Losses should be tracked   |
  :                       :                  : explicitly or with Keras   :
  :                       :                  : APIs, for example,         :
  :                       :                  : [add_loss][add_loss],      :
  :                       :                  : instead of via collections :
  | `reduction`           | `reduction`      | In constructor. Value of   |
  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :
  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :
  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :
  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :
  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :
  : : : `tf.keras.losses.Reduction.SUM`,                                  :
  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :
  : : : used other value for `reduction`, including the default value     :
  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :
  : : : no directly corresponding value. Please modify the loss           :
  : : : implementation manually.                                          :

  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss


  #### Before & After Usage Example

  Before:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> weights = [0.3, 0.7]
  >>> smoothing = 0.2
  >>> tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights,
  ...   label_smoothing=smoothing).numpy()
  0.57618

  After:

  >>> cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
  ...   label_smoothing=smoothing)
  >>> cce(y_true, y_pred, sample_weight=weights).numpy()
  0.57618

  @end_compatibility
  """
    if onehot_labels is None:
        raise ValueError('Argument `onehot_labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'softmax_cross_entropy_loss', (logits, onehot_labels, weights)) as scope:
        logits = ops.convert_to_tensor(logits)
        onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
        logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
        if label_smoothing > 0:
            num_classes = math_ops.cast(array_ops.shape(onehot_labels)[-1], logits.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            onehot_labels = onehot_labels * smooth_positives + smooth_negatives
        onehot_labels = array_ops.stop_gradient(onehot_labels, name='labels_stop_gradient')
        losses = nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits, name='xentropy')
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)