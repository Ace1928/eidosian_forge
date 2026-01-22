from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops.gen_state_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def init_variable(v, init, name='init'):
    """Initializes variable with "init".

  This op does the following:
  if init is a Tensor, v = init
  if callable(init): v = init(VariableShape(v), v.dtype)

  Args:
    v: Variable to initialize
    init: Tensor to assign to v,
      Or an object convertible to Tensor e.g. nparray,
      Or an Initializer that generates a tensor given the shape and type of v.
      An "Initializer" is a callable that returns a tensor that "v" should be
      set to. It will be called as init(shape, dtype).
    name: Optional name for the op.

  Returns:
    The operation that initializes v.
  """
    with ops.name_scope(None, v.op.name + '/', [v, init]):
        with ops.name_scope(name) as scope:
            with ops.colocate_with(v):
                if callable(init):
                    assert v.get_shape().is_fully_defined(), 'Variable shape unknown.'
                    value = init(v.get_shape().as_list(), v.dtype.base_dtype)
                    value = ops.convert_to_tensor(value, name='value')
                    return gen_state_ops.assign(v, value, name=scope)
                else:
                    init = ops.convert_to_tensor(init, name='init')
                    return gen_state_ops.assign(v, init, name=scope)