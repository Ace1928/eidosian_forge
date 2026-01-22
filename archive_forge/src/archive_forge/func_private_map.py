import re
from tensorflow.python.util import tf_inspect
@property
def private_map(self):
    """A map from parents to symbols that should not be included at all.

    This map can be edited, but it should not be edited once traversal has
    begun.

    Returns:
      The map marking symbols to not include.
    """
    return self._private_map