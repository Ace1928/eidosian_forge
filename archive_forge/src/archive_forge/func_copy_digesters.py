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
def copy_digesters(digesters):
    """Returns copy of provided digesters since deepcopying doesn't work."""
    result = {}
    for hash_algorithm in digesters:
        result[hash_algorithm] = digesters[hash_algorithm].copy()
    return result