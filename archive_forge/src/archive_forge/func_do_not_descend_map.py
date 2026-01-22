import re
from tensorflow.python.util import tf_inspect
@property
def do_not_descend_map(self):
    """A map from parents to symbols that should not be descended into.

    This map can be edited, but it should not be edited once traversal has
    begun.

    Returns:
      The map marking symbols to not explore.
    """
    return self._do_not_descend_map