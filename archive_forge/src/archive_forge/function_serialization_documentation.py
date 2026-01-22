from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
Wraps the concrete function if it uses cached read tensors.

  This function creates a new concrete function that captures variables
  instead of the cached read tensors.

  Args:
    concrete_function: A Concrete function that maybe captures cached read
      tensors.

  Returns:
    A concrete function that wraps the original concrete function, which
    captures variables instead. If the original function did not capture any
    cached values, then the function is not wrapped and the original object is
    returned.
  