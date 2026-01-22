from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
def track_numerical_issues(exc_info):
    """No tracking for external library.

  Args:
    exc_info: Output from `sys.exc_info` (type, value, traceback)
  """
    del exc_info