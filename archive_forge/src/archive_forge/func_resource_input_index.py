from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import gradients_util
from tensorflow.python.util import keras_deps
from tensorflow.python.util import tf_contextlib
def resource_input_index(tensor_name, input_names, node_defs, functions):
    """Returns the index of the input corresponding to `tensor_name`.

  This method is used to find the corresponding index of an arbitrary resource
  tensor in a function (the function could be a loop body). We assume that
  resource handles are never created in functions, so that every resource
  tensor can be traced back to a function input.

  The awkward signature of this method is to make it work with both FuncGraphs
  and FunctionDefs. This is so we can recurse on function call ops without
  building the corresponding FuncGraph (note that even if a FuncGraph for a
  FunctionDef already exists, the input/output/node names may have been
  changed when the FuncGraph was serialized to the FunctionDef, which makes it
  unusable with this algorithm).

  Args:
    tensor_name: the name of the resource tensor to be resolved to an input.
    input_names: a list of the names of all inputs to the function.
    node_defs: a dict mapping op name -> NodeDef for every op in the function.
    functions: a dict mapping function name -> AtomicFunction.

  Returns:
    The index into input_names corresponding to `tensor_name`.
  """
    while tensor_name not in input_names:
        parts = tensor_name.split(':')
        if len(parts) == 3:
            op_name, _, output_idx = parts
        elif len(parts) == 2:
            op_name, output_idx = parts
        else:
            assert len(parts) == 1
            op_name = parts[0]
            output_idx = 0
            tensor_name = '%s:%d' % (tensor_name, output_idx)
            if tensor_name in input_names:
                break
        output_idx = int(output_idx)
        node_def = node_defs[op_name]

        def _extract_input_index(function_attribute_name):
            func_name = node_def.attr[function_attribute_name].func.name
            fdef = functions[func_name].cached_definition
            output_arg_name = fdef.signature.output_arg[output_idx].name
            output_tensor_name = fdef.ret[output_arg_name]
            return resource_input_index(output_tensor_name, [arg.name for arg in fdef.signature.input_arg], {ndef.name: ndef for ndef in fdef.node_def}, functions)
        if node_def.op in ('Identity', 'While'):
            tensor_name = node_def.input[output_idx]
        elif node_def.op in ('PartitionedCall', 'StatefulPartitionedCall'):
            tensor_name = node_def.input[_extract_input_index('f')]
        elif node_def.op in ('If', 'StatelessIf'):
            input_index = _extract_input_index('then_branch')
            if input_index != _extract_input_index('else_branch'):
                raise AssertionError('Expected cond branches ({} op) to each have the same input->output mapping of resources.'.format(node_def.op))
            tensor_name = node_def.input[input_index + 1]
        else:
            raise ValueError('Taking gradient of a while loop which creates a resource in its body is not supported: %s (%s)' % (op_name, node_def.op))
    return input_names.index(tensor_name)