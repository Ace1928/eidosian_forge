import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def func_graph_from_py_func(name, python_func, args, kwargs, signature=None, func_graph=None, add_control_dependencies=True, arg_names=None, op_return_value=None, collections=None, capture_by_value=None, create_placeholders=True):
    """Returns a `FuncGraph` generated from `python_func`.

  Args:
    name: an identifier for the function.
    python_func: the Python function to trace.
    args: the positional args with which the Python function should be called;
      ignored if a signature is provided.
    kwargs: the keyword args with which the Python function should be called;
      ignored if a signature is provided.
    signature: a possibly nested sequence of `TensorSpecs` specifying the shapes
      and dtypes of the arguments. When a signature is provided, `args` and
      `kwargs` are ignored, and `python_func` is traced with Tensors conforming
      to `signature`. If `None`, the shapes and dtypes are inferred from the
      inputs.
    func_graph: Optional. An instance of FuncGraph. If provided, we will use
      this graph else a new one is built and returned.
    add_control_dependencies: If True, automatically adds control dependencies
      to ensure program order matches execution order and stateful ops always
      execute.
    arg_names: Optional list of argument names, used to give input placeholders
      recognizable names.
    op_return_value: Optional. A Tensor. If set and `python_func` returns
      Operations, those return values will be replaced with this value. If not
      set, returning an Operation triggers an error.
    collections: a dictionary of collections this FuncGraph should start with.
      If not specified (None), the FuncGraph will read (but not write to) the
      outer graph's collections that are not allowlisted, and both read and
      write to the outer graph's collections that are allowlisted. The current
      allowlisted collections are the global variables, the local variables, and
      the trainable variables. Defaults to None.
    capture_by_value: An optional boolean. If True, the func graph will capture
      Variables by value instead of reference. By default inherit from outer
      graphs, and failing that will default to False.
    create_placeholders: An optional boolean. If True, then func graph will
      create placeholders for the inputs as graph ops. If False, the input args
      and kwargs will be treated as the input placeholders.

  Returns:
    A FuncGraph.

  Raises:
    TypeError: If any of `python_func`'s return values is neither `None`, a
      `Tensor` or a `tf.experimental.ExtensionType`.
  """
    if op_return_value is not None:
        assert isinstance(op_return_value, tensor_lib.Tensor), op_return_value
    if func_graph is None:
        func_graph = FuncGraph(name, collections=collections, capture_by_value=capture_by_value)
    assert isinstance(func_graph, FuncGraph)
    if add_control_dependencies:
        deps_control_manager = auto_control_deps.AutomaticControlDependencies()
    else:
        deps_control_manager = ops.NullContextmanager()
    with func_graph.as_default(), deps_control_manager as deps_ctx:
        current_scope = variable_scope.get_variable_scope()
        default_use_resource = current_scope.use_resource
        current_scope.set_use_resource(True)
        if signature is not None:
            args = signature
            kwargs = {}
        if create_placeholders:
            func_args, func_kwargs = _create_placeholders(args, kwargs, arg_names)
        else:
            func_args, func_kwargs = (args, kwargs)
        input_trace_types = trace_type.from_value([func_args, func_kwargs])
        func_graph.inputs = input_trace_types._to_tensors([func_args, func_kwargs])
        func_graph._watched_variables = object_identity.ObjectIdentityWeakSet()
        for arg in func_graph.inputs:
            if arg.dtype == dtypes.resource:
                func_graph._resource_tensor_inputs.add(arg)
        signature_context = trace_type.InternalTracingContext()
        func_graph.structured_input_signature = (convert_structure_to_signature(func_args, arg_names, signature_context=signature_context), convert_structure_to_signature(func_kwargs, signature_context=signature_context))
        func_args_before = nest.pack_sequence_as(func_args, nest.flatten(func_args, expand_composites=True), expand_composites=True)
        func_kwargs_before = nest.pack_sequence_as(func_kwargs, nest.flatten(func_kwargs, expand_composites=True), expand_composites=True)

        def convert(x):
            """Converts a function output to a Tensor."""
            if x is None:
                return None
            if op_return_value is not None and isinstance(x, ops.Operation):
                with ops.control_dependencies([x]):
                    x = array_ops.identity(op_return_value)
            elif not isinstance(x, tensor_array_ops.TensorArray):
                try:
                    x = ops.convert_to_tensor_or_composite(x)
                except (ValueError, TypeError):
                    raise TypeError(f'To be compatible with tf.function, Python functions must return zero or more Tensors or ExtensionTypes or None values; in compilation of {str(python_func)}, found return value of type {type(x).__name__}, which is not a Tensor or ExtensionType.')
            if add_control_dependencies:
                x = deps_ctx.mark_as_return(x)
            return x
        _, original_func = tf_decorator.unwrap(python_func)
        func_outputs = python_func(*func_args, **func_kwargs)
        func_outputs = variable_utils.convert_variables_to_tensors(func_outputs)
        func_outputs = nest.map_structure(convert, func_outputs, expand_composites=True)
        func_args = nest.pack_sequence_as(func_args, nest.flatten(func_args, expand_composites=True), expand_composites=True)
        func_kwargs = nest.pack_sequence_as(func_kwargs, nest.flatten(func_kwargs, expand_composites=True), expand_composites=True)
        check_func_mutation(func_args_before, func_kwargs_before, func_args, func_kwargs, original_func)
        current_scope.set_use_resource(default_use_resource)
        inputs = []
        for arg in composite_tensor_utils.flatten_with_variables([func_args, func_kwargs]):
            if isinstance(arg, resource_variable_ops.BaseResourceVariable):
                capture = func_graph._function_captures.pop(id(arg.handle), False)
                assert len(capture) >= 2
                resource_placeholder = capture[1]
                if resource_placeholder is None:
                    continue
                inputs.append(resource_placeholder)
            elif isinstance(arg, tensor_lib.Tensor):
                inputs.append(arg)
        func_graph.inputs = inputs + func_graph.internal_captures + nest.flatten(func_graph.deferred_internal_captures, expand_composites=True)
        func_graph.structured_outputs = func_outputs
        func_graph.outputs.extend((func_graph.capture(x) for x in flatten(func_graph.structured_outputs) if x is not None))
        func_graph.variables = func_graph._watched_variables
    if add_control_dependencies:
        func_graph.control_outputs.extend(deps_control_manager.ops_which_must_run)
        func_graph.collective_manager_ids_used = deps_control_manager.collective_manager_ids_used
    return func_graph