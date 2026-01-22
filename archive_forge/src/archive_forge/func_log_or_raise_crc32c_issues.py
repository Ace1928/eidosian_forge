from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import struct
import textwrap
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def log_or_raise_crc32c_issues(warn_for_always=True):
    """Informs user about slow hashing if requested.

  Args:
    warn_for_always (bool): User may not want to see a warning about slow hashes
      if they have the "always check hashes" property set because (1) they
      intentionally set a property and (2) it could duplicate a warning in
      FileDownloadTask.

  Raises:
    errors.Error: IF_FAST_ELSE_FAIL set, and CRC32C binary not present. See
      error message for more details.
  """
    if check_if_will_use_fast_crc32c(install_if_missing=True):
        return
    check_hashes = properties.VALUES.storage.check_hashes.Get()
    if check_hashes == properties.CheckHashes.ALWAYS.value and warn_for_always:
        log.warning(_SLOW_HASH_CHECK_WARNING)
    elif check_hashes == properties.CheckHashes.IF_FAST_ELSE_SKIP.value:
        log.warning(_NO_HASH_CHECK_WARNING)
    elif check_hashes == properties.CheckHashes.IF_FAST_ELSE_FAIL.value:
        raise errors.Error(_NO_HASH_CHECK_ERROR)