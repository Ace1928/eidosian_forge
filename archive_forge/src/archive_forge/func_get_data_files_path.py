import os as _os
import sys as _sys
from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['resource_loader.get_data_files_path'])
def get_data_files_path():
    """Get a direct path to the data files colocated with the script.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
    return _os.path.dirname(_inspect.getfile(_sys._getframe(1)))