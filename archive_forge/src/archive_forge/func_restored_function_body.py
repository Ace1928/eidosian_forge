import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def restored_function_body(*args, **kwargs):
    """Calls a restored function or raises an error if no matching function."""
    if not saved_function.concrete_functions:
        raise ValueError('Found zero restored functions for caller function.')
    inputs = (args, kwargs)
    for allow_conversion in [False, True]:
        for function_name in saved_function.concrete_functions:
            function = concrete_functions[function_name]
            if any([inp is None for inp in function.captured_inputs]):
                raise ValueError('Looks like you are trying to run a loaded non-Keras model that was trained using tf.distribute.experimental.ParameterServerStrategy with variable partitioning, which is not currently supported. Try using Keras to define your model if possible.')
            if _concrete_function_callable_with(function, inputs, allow_conversion):
                return _call_concrete_function(function, inputs)
    signature_descriptions = []

    def _pretty_format_positional(positional):
        return 'Positional arguments ({} total):\n    * {}'.format(len(positional), '\n    * '.join((pprint.pformat(a) for a in positional)))
    for index, function_name in enumerate(saved_function.concrete_functions):
        concrete_function = concrete_functions[function_name]
        positional, keyword = concrete_function.structured_input_signature
        signature_descriptions.append('Option {}:\n  {}\n  Keyword arguments: {}'.format(index + 1, _pretty_format_positional(positional), keyword))
    raise ValueError(f'Could not find matching concrete function to call loaded from the SavedModel. Got:\n  {_pretty_format_positional(args)}\n  Keyword arguments: {kwargs}\n\n Expected these arguments to match one of the following {len(saved_function.concrete_functions)} option(s):\n\n{(chr(10) + chr(10)).join(signature_descriptions)}')