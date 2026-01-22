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
def raise_if_source_and_destination_not_valid_for_preserve_posix(source_url, destination_url):
    """Logs errors and returns bool indicating if transfer is valid for POSIX."""
    if isinstance(source_url, storage_url.FileUrl) and source_url.is_stream:
        raise errors.InvalidUrlError('Cannot preserve POSIX data from pipe: {}'.format(source_url))
    if isinstance(destination_url, storage_url.FileUrl) and destination_url.is_stream:
        raise errors.InvalidUrlError('Cannot write POSIX data to pipe: {}'.format(destination_url))
    if isinstance(source_url, storage_url.CloudUrl) and isinstance(destination_url, storage_url.CloudUrl):
        raise errors.InvalidUrlError('Cannot preserve POSIX data for cloud-to-cloud copies')