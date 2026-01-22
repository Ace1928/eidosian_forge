from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def does_data_match_checksum(data, crc32c_checksum):
    """Checks if checksum for the data matches the supplied checksum.

  Args:
    data (bytes): Bytes over which the checksum should be calculated.
    crc32c_checksum (int): Checksum against which data's checksum will be
      compared.

  Returns:
    True iff both checksums match.
  """
    crc = get_crc32c()
    crc.update(six.ensure_binary(data))
    return get_checksum(crc) == crc32c_checksum