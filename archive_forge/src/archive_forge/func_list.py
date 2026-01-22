import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def list(self):
    """Lists registered items.

    Returns:
      A list of names of registered objects.
    """
    return self._registry.keys()