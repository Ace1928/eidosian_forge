import os as _os
import sys as _sys
from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['resource_loader.get_root_dir_with_all_resources'])
def get_root_dir_with_all_resources():
    """Get a root directory containing all the data attributes in the build rule.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary. Falls back to returning the same as get_data_files_path if it
    fails to detect a bazel runfiles directory.
  """
    script_dir = get_data_files_path()
    directories = [script_dir]
    data_files_dir = ''
    while True:
        candidate_dir = directories[-1]
        current_directory = _os.path.basename(candidate_dir)
        if '.runfiles' in current_directory:
            if len(directories) > 1:
                data_files_dir = directories[-2]
            break
        else:
            new_candidate_dir = _os.path.dirname(candidate_dir)
            if new_candidate_dir == candidate_dir:
                break
            else:
                directories.append(new_candidate_dir)
    return data_files_dir or script_dir