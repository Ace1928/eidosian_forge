import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.sigmoid_cross_entropy_with_logits'])
@dispatch.add_dispatch_support
def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    """See sigmoid_cross_entropy_with_logits_v2."""
    nn_ops._ensure_xent_args('sigmoid_cross_entropy_with_logits', labels, logits)
    with ops.name_scope(name, 'logistic_loss', [logits, labels]) as name:
        logits = ops.convert_to_tensor(logits, name='logits')
        labels = ops.convert_to_tensor(labels, name='labels')
        try:
            labels.get_shape().assert_is_compatible_with(logits.get_shape())
        except ValueError:
            raise ValueError(f'`logits` and `labels` must have the same shape, received ({logits.get_shape()} vs {labels.get_shape()}).')
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = logits >= zeros
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)
        return math_ops.add(relu_logits - logits * labels, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=name)