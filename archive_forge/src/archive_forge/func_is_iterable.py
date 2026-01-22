from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def is_iterable(obj):
    """Return true if the object is iterable."""
    if isinstance(obj, tensor_lib.Tensor):
        return False
    try:
        _ = iter(obj)
    except Exception:
        return False
    return True