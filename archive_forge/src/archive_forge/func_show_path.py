from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def show_path(from_op, tensors, sources):
    """Find one path from `from_op` to any of `tensors`, ignoring `sources`.

  Args:
    from_op: A `tf.Operation`.
    tensors: A `tf.Operation`, a `tf.Tensor`, or a list thereof.
    sources: A list of `tf.Tensor`.

  Returns:
    A python string containing the path, or "??" if none is found.
  """
    if isinstance(from_op, tensor_lib.Tensor):
        from_op = from_op.op
    if not isinstance(tensors, list):
        tensors = [tensors]
    final_ops = [_as_operation(tensor) for tensor in tensors]
    visited_ops = set((x.op for x in sources))
    ops_to_visit = list(final_ops)
    some_op_output = {}
    while ops_to_visit:
        op = ops_to_visit.pop()
        if op in visited_ops:
            continue
        visited_ops.add(op)
        if op == from_op:
            path_op = op
            path = [path_op]
            while path_op not in final_ops:
                path_op = some_op_output[path_op]
                path.append(path_op)
            return ' <- '.join(('%s (%s)' % (x.name, x.type) for x in reversed(path)))
        else:
            for inp in graph_inputs(op):
                if inp not in visited_ops and inp not in sources:
                    some_op_output[inp] = op
                    ops_to_visit.append(inp)
    return '??'