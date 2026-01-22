from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def read_directory_changes(handle, recursive):
    """Read changes to the directory using the specified directory handle.

    http://timgolden.me.uk/pywin32-docs/win32file__ReadDirectoryChangesW_meth.html
    """
    event_buffer = ctypes.create_string_buffer(BUFFER_SIZE)
    nbytes = ctypes.wintypes.DWORD()
    try:
        ReadDirectoryChangesW(handle, ctypes.byref(event_buffer), len(event_buffer), recursive, WATCHDOG_FILE_NOTIFY_FLAGS, ctypes.byref(nbytes), None, None)
    except WindowsError as e:
        if e.winerror == ERROR_OPERATION_ABORTED:
            return ([], 0)
        raise e
    try:
        int_class = long
    except NameError:
        int_class = int
    return (event_buffer.raw, int_class(nbytes.value))