from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_ops
def f_wrapper(*tensor_args):
    f_args = tuple((tensor_args[tensor_args_idx[i]] if arg_is_tensor[i] else a for i, a in enumerate(args)))
    f_kwargs = {k: tensor_args[tensor_args_idx[k]] if kwarg_is_tensor[k] else kwargs[k] for i, k in enumerate(kwarg_keys)}
    f(*f_args, **f_kwargs)
    return 1