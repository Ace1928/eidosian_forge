import errno
import hashlib
import importlib
import os
import platform
import sys
from tensorflow.python.client import pywrap_tf_session as py_tf
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.register_filesystem_plugin')
def register_filesystem_plugin(plugin_location):
    """Loads a TensorFlow FileSystem plugin.

  Args:
    plugin_location: Path to the plugin. Relative or absolute filesystem plugin
      path to a dynamic library file.

  Returns:
    None

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
    if os.path.exists(plugin_location):
        py_tf.TF_RegisterFilesystemPlugin(plugin_location)
    else:
        raise OSError(errno.ENOENT, 'The file to load file system plugin from does not exist.', plugin_location)