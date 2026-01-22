from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def get_symlink_placeholder_file(source_path):
    """Creates a placholder file for the given symlink.

  The placeholder will be created in the directory specified by the
  symlink_placeholder_directory property, and its content will be the path
  targeted by the given symlink.

  Args:
    source_path: The path to an existing symlink for which a placeholder should
      be created.

  Returns:
    The path to the placeholder file that was created to represent the given
    symlink.
  """
    placeholder_path = get_symlink_placeholder_path(source_path)
    with files.BinaryFileWriter(placeholder_path) as placeholder_writer:
        placeholder_writer.write(os.readlink(source_path).encode('utf-8'))
    return placeholder_path