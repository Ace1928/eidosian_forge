import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def padding_fifo_queue(component_types, shapes=[], capacity: int=-1, container: str='', shared_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """A queue that produces elements in first-in first-out order.

  Variable-size shapes are allowed by setting the corresponding shape dimensions
  to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
  size of any given element in the minibatch.  See below for details.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types.
      Shapes of fixed rank but variable size are allowed by setting
      any shape dimension to -1.  In this case, the inputs' shape may vary along
      the given dimension, and DequeueMany will pad the given dimension with
      zeros up to the maximum shape of all elements in the given batch.
      If the length of this attr is 0, different queue elements may have
      different ranks and shapes, but only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("padding_fifo_queue op does not support eager execution. Arg 'handle' is a ref.")
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'padding_fifo_queue' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if shapes is None:
        shapes = []
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'padding_fifo_queue' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    if capacity is None:
        capacity = -1
    capacity = _execute.make_int(capacity, 'capacity')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('PaddingFIFOQueue', component_types=component_types, shapes=shapes, capacity=capacity, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('component_types', _op.get_attr('component_types'), 'shapes', _op.get_attr('shapes'), 'capacity', _op._get_attr_int('capacity'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('PaddingFIFOQueue', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result