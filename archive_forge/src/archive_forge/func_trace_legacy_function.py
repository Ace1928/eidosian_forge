import warnings
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import function_utils
from tensorflow.python.util import variable_utils
def trace_legacy_function(defun_kwargs):

    @function.Defun(*structure.get_flat_tensor_types(self._input_structure), **defun_kwargs)
    def wrapped_fn(*args):
        ret = wrapper_helper(*args)
        return structure.to_tensor_list(self._output_structure, ret)
    return lambda: wrapped_fn