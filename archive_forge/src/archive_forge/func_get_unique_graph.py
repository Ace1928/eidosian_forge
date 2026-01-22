from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def get_unique_graph(tops, check_types=None, none_if_empty=False):
    """Return the unique graph used by the all the elements in tops.

  Args:
    tops: iterable of elements to check (usually a list of tf.Operation and/or
      tf.Tensor). Or a tf.Graph.
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  """
    if isinstance(tops, ops.Graph):
        return tops
    if not is_iterable(tops):
        raise TypeError('{} is not iterable'.format(type(tops)))
    if check_types is None:
        check_types = (ops.Operation, tensor_lib.Tensor)
    elif not is_iterable(check_types):
        check_types = (check_types,)
    g = None
    for op in tops:
        if not isinstance(op, check_types):
            raise TypeError('Expected a type in ({}), got: {}'.format(', '.join([str(t) for t in check_types]), type(op)))
        if g is None:
            g = op.graph
        elif g._graph_key != op.graph._graph_key:
            raise ValueError('Operation {} does not belong to given graph'.format(op))
    if g is None and (not none_if_empty):
        raise ValueError("Can't find the unique graph of an empty list")
    return g