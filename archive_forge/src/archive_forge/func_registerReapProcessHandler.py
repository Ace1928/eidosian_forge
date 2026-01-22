from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def registerReapProcessHandler(pid, process):
    """
    Register a process handler for the given pid, in case L{reapAllProcesses}
    is called.

    @param pid: the pid of the process.
    @param process: a process handler.
    """
    if pid in reapProcessHandlers:
        raise RuntimeError('Try to register an already registered process.')
    try:
        auxPID, status = os.waitpid(pid, os.WNOHANG)
    except BaseException:
        log.msg(f'Failed to reap {pid}:')
        log.err()
        if pid is None:
            return
        auxPID = None
    if auxPID:
        process.processEnded(status)
    else:
        reapProcessHandlers[pid] = process