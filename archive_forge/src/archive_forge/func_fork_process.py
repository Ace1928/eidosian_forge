from __future__ import (absolute_import, division, print_function)
import glob
import os
import pickle
import platform
import select
import shlex
import subprocess
import traceback
from ansible.module_utils.six import PY2, b
from ansible.module_utils.common.text.converters import to_bytes, to_text
def fork_process():
    """
    This function performs the double fork process to detach from the
    parent process and execute.
    """
    pid = os.fork()
    if pid == 0:
        fd = os.open(os.devnull, os.O_RDWR)
        for num in range(3):
            if fd != num:
                os.dup2(fd, num)
        if fd not in range(3):
            os.close(fd)
        pid = os.fork()
        if pid > 0:
            os._exit(0)
        sid = os.setsid()
        if sid == -1:
            raise Exception('Unable to detach session while daemonizing')
        os.chdir('/')
        pid = os.fork()
        if pid > 0:
            os._exit(0)
    return pid