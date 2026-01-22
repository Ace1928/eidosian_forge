import itertools
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import resource_variable_ops
def function_def_to_graph_def(fdef, input_shapes=None, include_library_functions=False):
    """Convert a FunctionDef to a GraphDef.

  Steps:
  1. Creates placeholder nodes corresponding to inputs in
     `FunctionDef.signature.input_arg`.
  2. Adds NodeDefs in `FunctionDef.node_def` to `GraphDef.node`.
  3. Renames inputs of all nodes to use the convention of GraphDef instead of
     FunctionDef. See comment on `FunctionDef.node_def` on how the tensor naming
     in FunctionDefs is different from GraphDefs.

  Args:
    fdef: FunctionDef.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. If specified, its length must match length of
      `fdef.signature.input_arg`. If a shape is None, the corresponding input
      placeholder will have unknown shape.
    include_library_functions: Optional. If enabled, copy `fdef` and its
      nested `FunctionDef`s to the library functions of the returned `GraphDef`.
      In graph mode, the functions will be found from outer graph. In eager
      mode, the functions will be found from eager context.

  Returns:
    A tuple of (GraphDef, dict<string, string>). The dict contains a mapping
    from nested tensor names (in FunctionDef) to flattened names (in GraphDef).

  Raises:
    ValueError: If the length of input_shapes does not match the number of
      input_args or if the FunctionDef is invalid.
  """
    graph_def = graph_pb2.GraphDef()
    graph_def.versions.CopyFrom(versions_pb2.VersionDef(producer=versions.GRAPH_DEF_VERSION, min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER))
    default_graph = ops.get_default_graph()
    copied_functions = set()
    if input_shapes and len(input_shapes) != len(fdef.signature.input_arg):
        raise ValueError(f'Length of `input_shapes` must match the number of `input_arg`s in `fdef`. Got {len(input_shapes)} `input_shapes` and {len(fdef.signature.input_arg)} `input_arg`s.')
    for i, arg_def in enumerate(fdef.signature.input_arg):
        node_def = graph_def.node.add()
        node_def.name = arg_def.name
        node_def.op = 'Placeholder'
        node_def.attr['dtype'].type = arg_def.type
        if input_shapes and input_shapes[i] is not None:
            input_shape = input_shapes[i]
            if not isinstance(input_shape, tensor_shape_pb2.TensorShapeProto):
                input_shape = input_shape.as_proto()
            node_def.attr['shape'].shape.CopyFrom(input_shape)
        arg_attrs = fdef.arg_attr[i].attr
        for k in arg_attrs:
            if k == '_output_shapes':
                if arg_attrs[k].WhichOneof('value') == 'list':
                    node_def.attr['shape'].shape.CopyFrom(arg_attrs[k].list.shape[0])
                elif arg_attrs[k].WhichOneof('value') == 'shape':
                    node_def.attr['shape'].shape.CopyFrom(arg_attrs[k].shape)
            elif k.startswith('_'):
                node_def.attr[k].CopyFrom(arg_attrs[k])
    graph_def.node.extend(fdef.node_def)
    nested_to_flat_tensor_name = {}
    for arg_def in fdef.signature.input_arg:
        nested_to_flat_tensor_name[arg_def.name] = '{}:0'.format(arg_def.name)
        control_name = '^' + arg_def.name
        nested_to_flat_tensor_name[control_name] = control_name
    for node_def in fdef.node_def:
        graph = default_graph
        while True:
            f = graph._functions.get(node_def.op, None)
            if f is not None or not hasattr(graph, 'outer_graph'):
                break
            graph = graph.outer_graph
        if f is not None:
            fdef = f.cached_definition
            op_def = fdef.signature
            if node_def.op not in copied_functions:
                graph_def.library.function.add().CopyFrom(fdef)
                copied_functions.add(node_def.op)
                if getattr(f, 'grad_func_name', None):
                    grad_def = function_pb2.GradientDef()
                    grad_def.function_name = f.name
                    grad_def.gradient_func = f.grad_func_name
                    graph_def.library.gradient.extend([grad_def])
        else:
            op_def = default_graph.op_def_for_type(node_def.op)
        for attr in op_def.attr:
            if attr.type == 'func':
                fname = node_def.attr[attr.name].func.name
                if fname and (not is_function(fname, default_graph)):
                    raise ValueError(f'Function {fname} was not found. Please make sure the FunctionDef `fdef` is correct.')
                if include_library_functions:
                    copy_function_def_to_graph_def_recursively(fname, graph_def, copied_functions, default_graph)
            elif attr.type == 'list(func)':
                for fn in node_def.attr[attr.name].list.func:
                    fname = fn.name
                    if fname and (not is_function(fname, default_graph)):
                        raise ValueError(f'Function {fname} was not found. Please make sure the FunctionDef `fdef` is correct.')
                    if include_library_functions:
                        copy_function_def_to_graph_def_recursively(fname, graph_def, copied_functions, default_graph)
        flattened_index = 0
        for arg_def in op_def.output_arg:
            num_args = _get_num_args(arg_def, node_def)
            for i in range(num_args):
                nested_name = '{}:{}:{}'.format(node_def.name, arg_def.name, i)
                flat_name = '{}:{}'.format(node_def.name, flattened_index)
                nested_to_flat_tensor_name[nested_name] = flat_name
                flattened_index += 1
        control_name = '^' + node_def.name
        nested_to_flat_tensor_name[control_name] = control_name
    for node_def in graph_def.node:
        for i in range(len(node_def.input)):
            node_def.input[i] = nested_to_flat_tensor_name[node_def.input[i]]
    return (graph_def, nested_to_flat_tensor_name)