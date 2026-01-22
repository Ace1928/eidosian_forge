import errno
import math
import os
import sys
from typing import TYPE_CHECKING
from .. import _core, _subprocess
from .._sync import CapacityLimiter, Event
from .._threads import to_thread_run_sync
def sync_wait_reapable(pid: int) -> None:
    P_PID = 1
    WEXITED = 4
    if sys.platform == 'darwin':
        WNOWAIT = 32
    else:
        WNOWAIT = 16777216
    result = waitid_ffi.new('siginfo_t *')
    while waitid_cffi(P_PID, pid, result, WEXITED | WNOWAIT) < 0:
        got_errno = waitid_ffi.errno
        if got_errno == errno.EINTR:
            continue
        raise OSError(got_errno, os.strerror(got_errno))