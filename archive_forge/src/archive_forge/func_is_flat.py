import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def is_flat(outputs):
    """Checks if outputs is a flat structure.

    Following structures and values are considered flat:
    1) None
    2) A single object
    3) A list or tuple of Tensors/Operations

    The only structures that this function understands are sequences,
    dictionaries and types defined using the attrs library.  E.g. this means
    that if outputs contains a single user-defined Object, it is considered to
    be flat. Errors are raised later on if that Object cannot be converted to a
    Tensor.

  Args:
    outputs: Output from `computation` inside `xla.compile`.

  Returns:
    A boolean indicates whether outputs is flat.
  """
    if isinstance(outputs, collections_abc.Sequence):
        for o in outputs:
            if isinstance(o, collections_abc.Sequence) or isinstance(o, collections_abc.Mapping) or hasattr(o.__class__, '__attrs_attrs__'):
                return False
    if isinstance(outputs, collections_abc.Mapping):
        return False
    if hasattr(outputs.__class__, '__attrs_attrs__'):
        return False
    return True