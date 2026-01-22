from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
def convert_base_ten_to_base_eight_str(base_ten_int):
    """Takes base ten integer, converts to octal, and removes extra chars."""
    oct_string = oct(base_ten_int)[2:]
    permission_bytes = oct_string[-3:]
    return '0' * (3 - len(permission_bytes)) + permission_bytes