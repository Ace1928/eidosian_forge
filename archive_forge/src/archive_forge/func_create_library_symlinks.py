import gc as _gc
import importlib as _importlib
import os as _os
import platform as _platform
import sys as _sys
import warnings as _warnings
import pyarrow.lib as _lib
from pyarrow.lib import (BuildInfo, RuntimeInfo, set_timezone_db_path,
from pyarrow.lib import (null, bool_,
from pyarrow.lib import (Buffer, ResizableBuffer, foreign_buffer, py_buffer,
from pyarrow.lib import (MemoryPool, LoggingMemoryPool, ProxyMemoryPool,
from pyarrow.lib import (NativeFile, PythonFile,
from pyarrow._hdfsio import HdfsFile, have_libhdfs
from pyarrow.lib import (ChunkedArray, RecordBatch, Table, table,
from pyarrow.lib import (ArrowCancelled,
import pyarrow.hdfs as hdfs
from pyarrow.ipc import serialize_pandas, deserialize_pandas
import pyarrow.ipc as ipc
import pyarrow.types as types
from pyarrow.filesystem import FileSystem as _FileSystem
from pyarrow.filesystem import LocalFileSystem as _LocalFileSystem
from pyarrow.hdfs import HadoopFileSystem as _HadoopFileSystem
from pyarrow.util import _deprecate_api, _deprecate_class
from pyarrow.ipc import (Message, MessageReader, MetadataVersion,
def create_library_symlinks():
    """
    With Linux and macOS wheels, the bundled shared libraries have an embedded
    ABI version like libarrow.so.17 or libarrow.17.dylib and so linking to them
    with -larrow won't work unless we create symlinks at locations like
    site-packages/pyarrow/libarrow.so. This unfortunate workaround addresses
    prior problems we had with shipping two copies of the shared libraries to
    permit third party projects like turbodbc to build their C++ extensions
    against the pyarrow wheels.

    This function must only be invoked once and only when the shared libraries
    are bundled with the Python package, which should only apply to wheel-based
    installs. It requires write access to the site-packages/pyarrow directory
    and so depending on your system may need to be run with root.
    """
    import glob
    if _sys.platform == 'win32':
        return
    package_cwd = _os.path.dirname(__file__)
    if _sys.platform == 'linux':
        bundled_libs = glob.glob(_os.path.join(package_cwd, '*.so.*'))

        def get_symlink_path(hard_path):
            return hard_path.rsplit('.', 1)[0]
    else:
        bundled_libs = glob.glob(_os.path.join(package_cwd, '*.*.dylib'))

        def get_symlink_path(hard_path):
            return '.'.join((hard_path.rsplit('.', 2)[0], 'dylib'))
    for lib_hard_path in bundled_libs:
        symlink_path = get_symlink_path(lib_hard_path)
        if _os.path.exists(symlink_path):
            continue
        try:
            _os.symlink(lib_hard_path, symlink_path)
        except PermissionError:
            print('Tried creating symlink {}. If you need to link to bundled shared libraries, run pyarrow.create_library_symlinks() as root')