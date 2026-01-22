from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def get_consuming_ops(ts):
    """Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the consuming `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if ts cannot be converted to a list of `tf.Tensor`.
  """
    ts = make_list_of_t(ts, allow_graph=False)
    tops = []
    for t in ts:
        for op in t.consumers():
            if op not in tops:
                tops.append(op)
    return tops