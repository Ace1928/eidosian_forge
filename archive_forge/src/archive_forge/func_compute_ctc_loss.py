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