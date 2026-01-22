import binascii
import os
from posixpath import join as urljoin
import uuid
import six
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('io.gfile.remove')
def delete_file_v2(path):
    """Deletes the path located at 'path'.

  Args:
    path: string, a path

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    `NotFoundError` if the path does not exist.
  """
    _pywrap_file_io.DeleteFile(compat.path_to_bytes(path))