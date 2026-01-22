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
def run_if_setting_posix(posix_to_set, user_request_args, function, *args, **kwargs):
    """Useful for gating functions without repeating the below if statement."""
    if posix_to_set or (user_request_args and user_request_args.preserve_posix):
        return function(*args, **kwargs)
    return None