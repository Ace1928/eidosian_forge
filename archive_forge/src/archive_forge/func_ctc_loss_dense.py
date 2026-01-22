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
def ctc_loss_dense(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=0, name=None):
    """Computes CTC (Connectionist Temporal Classification) loss.

  This op implements the CTC loss as presented in (Graves et al., 2006),
  using the batched forward backward algorithm described in (Sim et al., 2017).

  Notes:
    Significant differences from `tf.compat.v1.nn.ctc_loss`:
      Supports GPU and TPU (`tf.compat.v1.nn.ctc_loss` supports CPU only):
        For batched operations, GPU and TPU are significantly faster than using
        `ctc_loss` on CPU.
        This implementation runs on CPU, but significantly slower than ctc_loss.
      Blank label is 0 rather num_classes - 1, unless overridden by blank_index.
      Logits and labels are dense arrays with padding rather than SparseTensor.
      The only mode supported is the same as:
        preprocess_collapse_repeated=False, ctc_merge_repeated=True
        To collapse labels, the caller can preprocess label sequence first.

    The dense implementation supports both CPU, GPU and TPU. A fast path is
    provided that significantly improves memory use for large vocabulary if the
    caller preprocesses label sequences to get unique label indices on the CPU
    (eg. in the data input pipeline) using ctc_ops.unique and simplifies this in
    the optional "unique" kwarg. This is especially useful for TPU and GPU but
    also works with if used on CPU.

  Args:
    labels: tensor of shape [batch_size, max_label_seq_length]
    logits: tensor of shape [frames, batch_size, num_labels], if
      logits_time_major == False, shape is [batch_size, frames, num_labels].
    label_length: tensor of shape [batch_size] Length of reference label
      sequence in labels.
    logit_length: tensor of shape [batch_size] Length of input sequence in
      logits.
    logits_time_major: (optional) If True (default), logits is shaped [time,
      batch, logits]. If False, shape is [batch, time, logits]
    unique: (optional) Unique label indices as computed by unique(labels). If
      supplied, enable a faster, memory efficient implementation on TPU.
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
      Improving the efficiency of forward-backward algorithm using batched
      computation in TensorFlow:
        [Sim et al., 2017](https://ieeexplore.ieee.org/document/8268944)
        ([pdf](http://bacchiani.net/resume/papers/ASRU2017.pdf))
  """
    with ops.name_scope(name, 'ctc_loss_dense', [logits, labels, label_length, logit_length]):
        logits = ops.convert_to_tensor(logits, name='logits')
        labels = ops.convert_to_tensor(labels, name='labels')
        label_length = ops.convert_to_tensor(label_length, name='label_length')
        logit_length = ops.convert_to_tensor(logit_length, name='logit_length')
        orig_dtype = logits.dtype
        if orig_dtype in (dtypes.float16, dtypes.bfloat16):
            logits = math_ops.cast(logits, dtypes.float32)
        if not logits_time_major:
            logits = array_ops.transpose(logits, perm=[1, 0, 2])
        if blank_index != 0:
            if blank_index < 0:
                blank_index += _get_dim(logits, 2)
            logits = array_ops.concat([logits[:, :, blank_index:blank_index + 1], logits[:, :, :blank_index], logits[:, :, blank_index + 1:]], axis=2)
            labels = array_ops.where(labels < blank_index, labels + 1, labels)
        args = [logits, labels, label_length, logit_length]
        if unique:
            unique_y, unique_idx = unique
            if blank_index != 0:
                unique_y = array_ops.where(unique_y < blank_index, unique_y + 1, unique_y)
                label_mask_len = math_ops.reduce_max(unique_idx, axis=1) + 1
                max_label_length = _get_dim(unique_y, 1)
                label_mask = array_ops.sequence_mask(label_mask_len, max_label_length)
                unique_y = array_ops.where(label_mask, unique_y, array_ops.zeros_like(unique_y))
            args.extend([unique_y, unique_idx])

        @custom_gradient.custom_gradient
        def compute_ctc_loss(logits_t, labels_t, label_length_t, logit_length_t, *unique_t):
            """Compute CTC loss."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(logits=logits_t, labels=labels_t, label_length=label_length_t, logit_length=logit_length_t)
            if unique_t:
                kwargs['unique'] = unique_t
            result = ctc_loss_and_grad(**kwargs)

            def grad(grad_loss):
                grad = [array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]]
                grad += [None] * (len(args) - len(grad))
                return grad
            return (result[0], grad)
        loss = compute_ctc_loss(*args)
        if orig_dtype in (dtypes.float16, dtypes.bfloat16):
            loss = math_ops.cast(loss, orig_dtype)
        return loss