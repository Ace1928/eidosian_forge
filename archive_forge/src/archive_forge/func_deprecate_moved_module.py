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
def deprecate_moved_module(deprecated_name, new_module, deletion_version):
    """Logs a warning when a module that has been moved to a new location is used.

  Copy the following code into the old module:

  ```
  import deprecation
  import new_module

  __getattr__ = deprecation.deprecate_moved_module(
      __name__, new_module, "2.9")  # adjust version number.
  ```

  Args:
    deprecated_name: Name of old module.
    new_module: Module to replace the old module.
    deletion_version: Version of TensorFlow in which the old module will be
      removed.

  Returns:
    A function that logs a warning and returns the symbol from the new module.
    Set this function as the module's `__getattr__`.
  """

    def getter(name):
        if getter not in _PRINTED_WARNING and _PRINT_DEPRECATION_WARNINGS:
            _PRINTED_WARNING[getter] = True
            _log_deprecation('Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.', deprecated_name, new_module.__name__, deletion_version)
        return getattr(new_module, name)
    return getter