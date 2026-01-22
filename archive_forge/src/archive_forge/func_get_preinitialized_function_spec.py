from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def get_preinitialized_function_spec(concrete_function):
    """Generates an unconstrained FunctionSpec from FunctionType."""
    if concrete_function.structured_input_signature is None or isinstance(concrete_function, wrap_function_lib.WrappedFunction):
        return None
    function_type = concrete_function.function_type
    if function_type is None:
        return None
    unconstrained_type = function_type_lib.FunctionType([function_type_lib.Parameter(p.name, p.kind, p.optional, None) for p in function_type.parameters.values()])
    default_values = {p.default for p in function_type.parameters.values() if p.optional}
    return function_type_utils.FunctionSpec(unconstrained_type, default_values, False, name=concrete_function.name)