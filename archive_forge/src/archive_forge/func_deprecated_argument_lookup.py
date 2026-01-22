import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def deprecated_argument_lookup(new_name, new_value, old_name, old_value):
    """Looks up deprecated argument name and ensures both are not used.

  Args:
    new_name: new name of argument
    new_value: value of new argument (or None if not used)
    old_name: old name of argument
    old_value: value of old argument (or None if not used)

  Returns:
    The effective argument that should be used.
  Raises:
    ValueError: if new_value and old_value are both non-null
  """
    if old_value is not None:
        if new_value is not None:
            raise ValueError(f"Cannot specify both '{old_name}' and '{new_name}'.")
        return old_value
    return new_value