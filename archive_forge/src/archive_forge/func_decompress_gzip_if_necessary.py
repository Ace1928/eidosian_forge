from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import os
import shutil
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def decompress_gzip_if_necessary(source_resource, gzipped_path, destination_path, do_not_decompress_flag=False, server_encoding=None):
    """Checks if file is elligible for decompression and decompresses if true.

  Args:
    source_resource (ObjectResource): May contain encoding metadata.
    gzipped_path (str): File path to unzip.
    destination_path (str): File path to write unzipped file to.
    do_not_decompress_flag (bool): User flag that blocks decompression.
    server_encoding (str|None): Server-reported `content-encoding` of file.

  Returns:
    (bool) True if file was successfully decompressed, else False.
  """
    content_encoding = getattr(source_resource.metadata, 'contentEncoding', '')
    if do_not_decompress_flag or not (content_encoding and 'gzip' in content_encoding.split(',') or (server_encoding and 'gzip' in server_encoding.split(','))):
        return False
    try:
        with gzip.open(gzipped_path, 'rb') as gzipped_file:
            with files.BinaryFileWriter(destination_path, create_path=True, convert_invalid_windows_characters=properties.VALUES.storage.convert_incompatible_windows_path_characters.GetBool()) as ungzipped_file:
                shutil.copyfileobj(gzipped_file, ungzipped_file)
        return True
    except OSError:
        os.remove(destination_path)
    return False