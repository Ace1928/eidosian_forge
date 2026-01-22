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
def trace_py_function(defun_kwargs):

    def unused(*args):
        ret = wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]
    func_name = defun_kwargs.pop('func_name', 'unused')
    tf_function = def_function.Function(python_function=unused, name=func_name, input_signature=structure.get_flat_tensor_specs(self._input_structure), autograph=False, experimental_attributes=defun_kwargs)
    _ = tf_function.get_concrete_function()

    def py_function_wrapper(*args):
        nested_args = structure.from_compatible_tensor_list(self._input_structure, args)
        if not _should_unpack(nested_args):
            nested_args = (nested_args,)
        ret = self._func(*nested_args)
        if _should_pack(ret):
            ret = tuple(ret)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

    @def_function.function(input_signature=structure.get_flat_tensor_specs(self._input_structure), autograph=False, experimental_attributes=defun_kwargs)
    def wrapped_fn(*args):
        return script_ops.eager_py_func(py_function_wrapper, args, structure.get_flat_tensor_types(self._output_structure))
    return wrapped_fn.get_concrete_function