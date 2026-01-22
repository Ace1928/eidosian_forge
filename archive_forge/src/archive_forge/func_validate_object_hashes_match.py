from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import enum
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
def validate_object_hashes_match(object_path, source_hash, destination_hash):
    """Confirms hashes match for copied objects.

  Args:
    object_path (str): URL of object being validated.
    source_hash (str): Hash of source object.
    destination_hash (str): Hash of destination object.

  Raises:
    HashMismatchError: Hashes are not equal.
  """
    if source_hash != destination_hash:
        raise errors.HashMismatchError('Source hash {} does not match destination hash {} for object {}.'.format(source_hash, destination_hash, object_path))