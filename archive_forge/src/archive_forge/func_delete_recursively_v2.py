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
@tf_export('io.gfile.rmtree')
def delete_recursively_v2(path):
    """Deletes everything under path recursively.

  Args:
    path: string, a path

  Raises:
    errors.OpError: If the operation fails.
  """
    _pywrap_file_io.DeleteRecursively(compat.path_to_bytes(path))