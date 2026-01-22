from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def update_custom_metadata_dict_with_symlink_attributes(metadata_dict, is_symlink):
    """Updates custom metadata_dict with symlink data."""
    if is_symlink:
        metadata_dict[resource_util.SYMLINK_METADATA_KEY] = 'true'
    elif resource_util.SYMLINK_METADATA_KEY in metadata_dict:
        del metadata_dict[resource_util.SYMLINK_METADATA_KEY]