from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def convert_to_gb(size_bytes: int) -> int:
    """Convert size_bytes to GB (2^30).

  Args:
    size_bytes: Size in bytes.

  Returns:
    Size in gibibytes.
  """
    return size_bytes // GB