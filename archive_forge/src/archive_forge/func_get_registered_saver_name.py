import collections
import re
from tensorflow.python.util import tf_inspect
def get_registered_saver_name(trackable):
    """Returns the name of the registered saver to use with Trackable."""
    try:
        return _saver_registry.get_registered_name(trackable)
    except LookupError:
        return None