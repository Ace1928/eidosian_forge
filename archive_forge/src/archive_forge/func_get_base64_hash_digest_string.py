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
def get_base64_hash_digest_string(hash_object):
    """Takes hashlib object and returns base64-encoded digest as string."""
    return get_base64_string(hash_object.digest())