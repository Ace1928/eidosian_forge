import collections
import weakref
from tensorflow.python.util import object_identity
def is_layer(obj):
    """Implicit check for Layer-like objects."""
    return hasattr(obj, '_is_layer') and (not isinstance(obj, type))