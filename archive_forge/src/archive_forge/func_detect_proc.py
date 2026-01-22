import io
import os
import re
import sys
from ._core import Process
def detect_proc():
    """Detect /proc filesystem style.

    This checks the /proc/{pid} directory for possible formats. Returns one of
    the following as str:

    * `stat`: Linux-style, i.e. ``/proc/{pid}/stat``.
    * `status`: BSD-style, i.e. ``/proc/{pid}/status``.
    """
    pid = os.getpid()
    for name in ('stat', 'status'):
        if os.path.exists(os.path.join('/proc', str(pid), name)):
            return name
    raise ProcFormatError('unsupported proc format')