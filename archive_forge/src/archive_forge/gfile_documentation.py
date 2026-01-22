from tensorflow.python.lib.io.file_io import copy as Copy
from tensorflow.python.lib.io.file_io import create_dir as MkDir
from tensorflow.python.lib.io.file_io import delete_file as Remove
from tensorflow.python.lib.io.file_io import delete_recursively as DeleteRecursively
from tensorflow.python.lib.io.file_io import file_exists as Exists
from tensorflow.python.lib.io.file_io import FileIO as _FileIO
from tensorflow.python.lib.io.file_io import get_matching_files as Glob
from tensorflow.python.lib.io.file_io import is_directory as IsDirectory
from tensorflow.python.lib.io.file_io import list_directory as ListDirectory
from tensorflow.python.lib.io.file_io import recursive_create_dir as MakeDirs
from tensorflow.python.lib.io.file_io import rename as Rename
from tensorflow.python.lib.io.file_io import stat as Stat
from tensorflow.python.lib.io.file_io import walk as Walk
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
File I/O wrappers without thread locking.

  Note, that this  is somewhat like builtin Python  file I/O, but
  there are  semantic differences to  make it more  efficient for
  some backing filesystems.  For example, a write  mode file will
  not  be opened  until the  first  write call  (to minimize  RPC
  invocations in network filesystems).
  