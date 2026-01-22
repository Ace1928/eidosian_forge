from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import tracker_file_util
def finalize_download(source_resource, temporary_file_path, final_file_path, do_not_decompress_flag=False, server_encoding=None, convert_symlinks=False):
    """Converts temporary file to final form.

  This may involve decompressing, renaming, and/or converting symlink
  placeholders to actual symlinks.

  Args:
    source_resource (ObjectResource): May contain encoding metadata.
    temporary_file_path (str): File path to unzip or rename.
    final_file_path (str): File path to write final file to.
    do_not_decompress_flag (bool): User flag that blocks decompression.
    server_encoding (str|None): Server-reported `content-encoding` of file.
    convert_symlinks (bool): Whether symlink placeholders should be converted to
      actual symlinks.

  Returns:
    (bool) True if file was decompressed, renamed, and/or converted to a
      symlink; False if file did not exist.
  """
    make_symlink = convert_symlinks and source_resource.is_symlink
    if make_symlink:
        decompress_or_rename_path = temporary_file_path + SYMLINK_TEMPORARY_PLACEHOLDER_SUFFIX
    else:
        decompress_or_rename_path = final_file_path
    decompress_or_rename_result = _decompress_or_rename_file(source_resource=source_resource, temporary_file_path=temporary_file_path, final_file_path=decompress_or_rename_path, do_not_decompress_flag=do_not_decompress_flag, server_encoding=server_encoding)
    if not decompress_or_rename_result:
        return False
    if make_symlink:
        symlink_util.create_symlink_from_temporary_placeholder(placeholder_path=decompress_or_rename_path, symlink_path=final_file_path)
        os.remove(decompress_or_rename_path)
    return decompress_or_rename_result