import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def eval_in_original_context(f, args, caller_fn_scope):
    """Executes the eval function in the context of a specified function."""
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)
    args = (args[0], ctx_frame.f_globals if len(args) < 2 else args[1], ctx_frame.f_locals if len(args) < 3 else args[2])
    return f(*args)