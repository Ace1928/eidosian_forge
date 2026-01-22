from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_export
def convert_to_tensor_v2(value, dtype=None, dtype_hint=None, name=None) -> tensor_lib.Tensor:
    """Converts the given `value` to a `Tensor`."""
    return tensor_conversion_registry.convert(value, dtype, name, preferred_dtype=dtype_hint)