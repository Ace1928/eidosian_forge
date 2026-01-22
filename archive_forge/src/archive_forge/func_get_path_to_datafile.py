import os as _os
import sys as _sys
from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['resource_loader.get_path_to_datafile'])
def get_path_to_datafile(path):
    """Get the path to the specified file in the data dependencies.

  The path is relative to tensorflow/

  Args:
    path: a string resource path relative to tensorflow/

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
    if runfiles:
        r = runfiles.Create()
        new_fpath = r.Rlocation(_os.path.abspath(_os.path.join('tensorflow', path)))
        if new_fpath is not None and _os.path.exists(new_fpath):
            return new_fpath
    old_filepath = _os.path.join(_os.path.dirname(_inspect.getfile(_sys._getframe(1))), path)
    return old_filepath