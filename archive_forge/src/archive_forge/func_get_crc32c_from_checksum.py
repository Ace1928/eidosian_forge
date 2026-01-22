from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def get_crc32c_from_checksum(checksum):
    """Returns Hashlib-like CRC32C object with a starting checksum.

  Args:
    checksum (int): CRC32C checksum representing the hash of processed data.

  Returns:
    google_crc32c.Checksum if google-crc32c is available or predefined.Crc
   instance from crcmod library. Both set to use initial checksum.
  """
    crc = get_crc32c()
    if IS_FAST_GOOGLE_CRC32C_AVAILABLE:
        crc._crc = checksum
    else:
        crc.crcValue = checksum
    return crc