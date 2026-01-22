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
@tf_export(v1=['gfile.DeleteRecursively'])
def delete_recursively(dirname):
    """Deletes everything under dirname recursively.

  Args:
    dirname: string, a path to a directory

  Raises:
    errors.OpError: If the operation fails.
  """
    delete_recursively_v2(dirname)