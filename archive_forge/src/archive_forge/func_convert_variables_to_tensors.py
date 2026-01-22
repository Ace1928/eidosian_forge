from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
def convert_variables_to_tensors(values):
    """Converts `ResourceVariable`s in `values` to `Tensor`s.

  If an object is a `CompositeTensor` and overrides its
  `_convert_variables_to_tensors` method, its `ResourceVariable` components
  will also be converted to `Tensor`s. Objects other than `ResourceVariable`s
  in `values` will be returned unchanged.

  Args:
    values: A nested structure of `ResourceVariable`s, or any other objects.

  Returns:
    A new structure with `ResourceVariable`s in `values` converted to `Tensor`s.
  """

    def _convert_resource_variable_to_tensor(x):
        if _pywrap_utils.IsResourceVariable(x):
            return ops.convert_to_tensor(x)
        elif isinstance(x, composite_tensor.CompositeTensor):
            return composite_tensor.convert_variables_to_tensors(x)
        else:
            return x
    return nest.map_structure(_convert_resource_variable_to_tensor, values)