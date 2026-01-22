import ctypes
import ctypes.util
from typing import Any, cast
from twisted.python.filepath import FilePath
def initializeModule(libc: ctypes.CDLL) -> None:
    """
    Initialize the module, checking if the expected APIs exist and setting the
    argtypes and restype for C{inotify_init}, C{inotify_add_watch}, and
    C{inotify_rm_watch}.
    """
    for function in ('inotify_add_watch', 'inotify_init', 'inotify_rm_watch'):
        if getattr(libc, function, None) is None:
            raise ImportError('libc6 2.4 or higher needed')
    libc.inotify_init.argtypes = []
    libc.inotify_init.restype = ctypes.c_int
    libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
    libc.inotify_rm_watch.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int