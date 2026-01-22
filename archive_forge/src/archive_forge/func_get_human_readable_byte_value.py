from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def get_human_readable_byte_value(byte_count, use_gsutil_style=False):
    """Generates a string for bytes with human-readable units.

  Args:
    byte_count (int): A number of bytes to format.
    use_gsutil_style (bool): Outputs units in the style of the gsutil CLI (e.g.
      gcloud -> "1.00kiB", gsutil -> "1 KiB").

  Returns:
    A string form of the number using size abbreviations (KiB, MiB, etc).
  """
    if use_gsutil_style:
        return _gsutil_format_byte_values(byte_count)
    return scaled_integer.FormatBinaryNumber(byte_count, decimal_places=2)