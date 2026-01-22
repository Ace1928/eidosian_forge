import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util
def spawn_main(pipe_handle, parent_pid=None, tracker_fd=None):
    """
    Run code specified by data received over pipe
    """
    assert is_forking(sys.argv), 'Not forking'
    if sys.platform == 'win32':
        import msvcrt
        import _winapi
        if parent_pid is not None:
            source_process = _winapi.OpenProcess(_winapi.SYNCHRONIZE | _winapi.PROCESS_DUP_HANDLE, False, parent_pid)
        else:
            source_process = None
        new_handle = reduction.duplicate(pipe_handle, source_process=source_process)
        fd = msvcrt.open_osfhandle(new_handle, os.O_RDONLY)
        parent_sentinel = source_process
    else:
        from . import resource_tracker
        resource_tracker._resource_tracker._fd = tracker_fd
        fd = pipe_handle
        parent_sentinel = os.dup(pipe_handle)
    exitcode = _main(fd, parent_sentinel)
    sys.exit(exitcode)