import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
def isClose(x, y, relative_tolerance):
    """Returns True if x is close to y given the relative tolerance or if x and y are both inf, both -inf, or both NaNs.

  This function does not distinguish between signalling and non-signalling NaN.

  Args:
    x: float value to be compared
    y: float value to be compared
    relative_tolerance: float. The allowable difference between the two values
      being compared is determined by multiplying the relative tolerance by the
      maximum of the two values. If this is not provided, then all floats are
      compared using string comparison.
  """
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) == math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= relative_tolerance * max(abs(x), abs(y))