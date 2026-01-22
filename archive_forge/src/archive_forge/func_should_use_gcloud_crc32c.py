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
def should_use_gcloud_crc32c(install_if_missing=False):
    """Returns True if gcloud-crc32c should be used and installs if needed.

  Args:
    install_if_missing (bool): Install gcloud-crc32c if not already present.

  Returns:
    True if the Go binary gcloud-crc32c should be used.
  """
    user_wants_gcloud_crc32c = properties.VALUES.storage.use_gcloud_crc32c.GetBool()
    if user_wants_gcloud_crc32c is False:
        return False
    if user_wants_gcloud_crc32c is None and crc32c.IS_FAST_GOOGLE_CRC32C_AVAILABLE:
        return False
    if install_if_missing:
        return _check_if_gcloud_crc32c_available(install_if_missing=True)
    return _is_gcloud_crc32c_installed()