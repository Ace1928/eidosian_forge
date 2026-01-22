import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.ctc_loss_v2'])
@dispatch.add_dispatch_support
def ctc_loss_v2(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None):
    """Computes CTC (Connectionist Temporal Classification) loss.

  This op implements the CTC loss as presented in (Graves et al., 2006).

  Notes:

  - Same as the "Classic CTC" in TensorFlow 1.x's tf.compat.v1.nn.ctc_loss
    setting of preprocess_collapse_repeated=False, ctc_merge_repeated=True
  - Labels may be supplied as either a dense, zero-padded tensor with a
    vector of label sequence lengths OR as a SparseTensor.
  - On TPU and GPU: Only dense padded labels are supported.
  - On CPU: Caller may use SparseTensor or dense padded labels but calling with
    a SparseTensor will be significantly faster.
  - Default blank label is 0 rather num_classes - 1, unless overridden by
    blank_index.

  Args:
    labels: tensor of shape [batch_size, max_label_seq_length] or SparseTensor
    logits: tensor of shape [frames, batch_size, num_labels], if
      logits_time_major == False, shape is [batch_size, frames, num_labels].
    label_length: tensor of shape [batch_size], None if labels is SparseTensor
      Length of reference label sequence in labels.
    logit_length: tensor of shape [batch_size] Length of input sequence in
      logits.
    logits_time_major: (optional) If True (default), logits is shaped [time,
      batch, logits]. If False, shape is [batch, time, logits]
    unique: (optional) Unique label indices as computed by
      ctc_unique_labels(labels).  If supplied, enable a faster, memory efficient
      implementation on TPU.
    blank_index: (optional) Set the class index to use for the blank label.
      Negative values will start from num_classes, ie, -1 will reproduce the
      ctc_loss behavior of using num_classes - 1 for the blank symbol. There is
      some memory/performance overhead to switching from the default of 0 as an
      additional shifted copy of the logits may be created.
    name: A name for this `Op`. Defaults to "ctc_loss_dense".

  Returns:
    loss: tensor of shape [batch_size], negative log probabilities.

  References:
      Connectionist Temporal Classification - Labeling Unsegmented Sequence Data
      with Recurrent Neural Networks:
        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)
        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))
  """
    if isinstance(labels, sparse_tensor.SparseTensor):
        if blank_index is None:
            raise ValueError('Argument `blank_index` must be provided when labels is a SparseTensor.')
        if blank_index < 0:
            blank_index += _get_dim(logits, 2)
        if blank_index != _get_dim(logits, 2) - 1:
            logits = array_ops.concat([logits[:, :, :blank_index], logits[:, :, blank_index + 1:], logits[:, :, blank_index:blank_index + 1]], axis=2)
            labels = sparse_tensor.SparseTensor(labels.indices, array_ops.where(labels.values < blank_index, labels.values, labels.values - 1), labels.dense_shape)
        return ctc_loss(labels=labels, inputs=logits, sequence_length=logit_length, time_major=logits_time_major)
    if blank_index is None:
        blank_index = 0
    return ctc_loss_dense(labels=labels, logits=logits, label_length=label_length, logit_length=logit_length, logits_time_major=logits_time_major, unique=unique, blank_index=blank_index, name=name)