from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def get_symlink_placeholder_path(source_path):
    """Returns a path suitable for storing a placeholder file for a symlink."""
    symlink_directory = _create_symlink_directory_if_needed()
    symlink_filename = tracker_file_util.get_hashed_file_name(tracker_file_util.get_delimiterless_file_path(source_path))
    tracker_file_util.raise_exceeds_max_length_error(symlink_filename)
    return os.path.join(symlink_directory, symlink_filename)