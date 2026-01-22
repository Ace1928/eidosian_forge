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
@tf_export('nn.ctc_greedy_decoder')
@dispatch.add_dispatch_support
def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True, blank_index=None):
    """Performs greedy decoding on the logits given in input (best path).

  Given a tensor as `inputs`, the `blank_index` parameter defines the class
  index of the blank symbol.

  For example:

  If `blank_index` is equal to 1:

  >>> inf = float("inf")
  >>> logits = tf.constant([[[   0., -inf, -inf],
  ...                        [ -2.3, -inf, -0.1]],
  ...                       [[ -inf, -0.5, -inf],
  ...                        [ -inf, -inf, -0.1]],
  ...                       [[ -inf, -inf, -inf],
  ...                        [ -0.1, -inf, -2.3]]])
  >>> seq_lens = tf.constant([2, 3])
  >>> outputs = tf.nn.ctc_greedy_decoder(
  ...     logits,
  ...     seq_lens,
  ...     blank_index=1)

  Notes:

  - Unlike `ctc_beam_search_decoder`, `ctc_greedy_decoder` considers blanks
    as regular elements when computing the probability of a sequence.
  - Default `blank_index` is `(num_classes - 1)`, unless overriden.

  If `merge_repeated` is `True`, merge repeated classes in output.
  This means that if consecutive logits' maximum indices are the same,
  only the first of these is emitted.  The sequence `A B B * B * B` (where '*'
  is the blank label) becomes

    * `A B B B` if `merge_repeated=True`.
    * `A B B B B` if `merge_repeated=False`.

  Args:
    inputs: 3-D `float` `Tensor` sized `[max_time, batch_size, num_classes]`.
      The logits.
    sequence_length: 1-D `int32` vector containing sequence lengths, having size
      `[batch_size]`.
    merge_repeated: Boolean.  Default: True.
    blank_index: (Optional). Default: `num_classes - 1`. Define the class index
      to use for the blank label. Negative values will start from num_classes,
      ie, -1 will reproduce the ctc_greedy_decoder behavior of using
      num_classes - 1 for the blank symbol, which corresponds to the default.

  Returns:
    A tuple `(decoded, neg_sum_logits)` where

    decoded: A single-element list. `decoded[0]`
      is an `SparseTensor` containing the decoded outputs s.t.:

      `decoded.indices`: Indices matrix `(total_decoded_outputs, 2)`.
        The rows store: `[batch, time]`.

      `decoded.values`: Values vector, size `(total_decoded_outputs)`.
        The vector stores the decoded classes.

      `decoded.dense_shape`: Shape vector, size `(2)`.
        The shape values are: `[batch_size, max_decoded_length]`

    neg_sum_logits: A `float` matrix `(batch_size x 1)` containing, for the
        sequence found, the negative of the sum of the greatest logit at each
        timeframe.
  """
    outputs = gen_ctc_ops.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=merge_repeated, blank_index=blank_index)
    decoded_ix, decoded_val, decoded_shape, log_probabilities = outputs
    return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)], log_probabilities)