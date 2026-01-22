from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
def standard_to_absl(level):
    """Converts an integer level from the standard value to the absl value.

  Args:
    level: int, a Python standard logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in absl logging.
  """
    if not isinstance(level, int):
        raise TypeError('Expect an int level, found {}'.format(type(level)))
    if level < 0:
        level = 0
    if level < STANDARD_DEBUG:
        return STANDARD_DEBUG - level + 1
    elif level < STANDARD_INFO:
        return ABSL_DEBUG
    elif level < STANDARD_WARNING:
        return ABSL_INFO
    elif level < STANDARD_ERROR:
        return ABSL_WARNING
    elif level < STANDARD_CRITICAL:
        return ABSL_ERROR
    else:
        return ABSL_FATAL