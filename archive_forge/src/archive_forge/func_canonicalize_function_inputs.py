import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def canonicalize_function_inputs(args, kwargs, function_type, default_values=None, is_pure=False):
    """Canonicalizes `args` and `kwargs`.

  Canonicalize the inputs to the Python function using FunctionType.
  In particular, we parse the varargs and kwargs that the
  original function was called with into a tuple corresponding to the
  Python function's positional (named) arguments and a dictionary
  corresponding to its kwargs.  Missing default arguments are added.

  If the FunctionType has an type constraints, then they are used to convert
  arguments to tensors; otherwise, any inputs containing numpy arrays are
  converted to tensors.


  Args:
    args: The varargs this object was called with.
    kwargs: The keyword args this function was called with.
    function_type: FunctionType to canonicalize against.
    default_values: Default values to use.
    is_pure: Force variable inputs to Tensors.

  Returns:
    A canonicalized ordering of the inputs, as well as full and filtered
    (Tensors and Variables only) versions of their concatenated flattened
    representations, represented by a tuple in the form (args, kwargs,
    flat_args, filtered_flat_args). Here: `args` is a full list of bound
    arguments, and `kwargs` contains only true keyword arguments, as opposed
    to named arguments called in a keyword-like fashion.

  Raises:
    ValueError: If a keyword in `kwargs` cannot be matched with a positional
      argument when an input signature is specified, or when the inputs
      do not conform to the input signature.
  """
    default_values = {} if not default_values else default_values
    if is_pure:
        args, kwargs = _convert_variables_to_tensors(args, kwargs)
    bound_arguments = bind_function_inputs(args, kwargs, function_type, default_values)
    return bound_arguments