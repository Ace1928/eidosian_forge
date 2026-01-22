from __future__ import division
import collections
import contextlib
import datetime
import functools
import os
import signal
import subprocess
import sys
import threading
import time
from . import _common
from ._common import AIX
from ._common import BSD
from ._common import CONN_CLOSE
from ._common import CONN_CLOSE_WAIT
from ._common import CONN_CLOSING
from ._common import CONN_ESTABLISHED
from ._common import CONN_FIN_WAIT1
from ._common import CONN_FIN_WAIT2
from ._common import CONN_LAST_ACK
from ._common import CONN_LISTEN
from ._common import CONN_NONE
from ._common import CONN_SYN_RECV
from ._common import CONN_SYN_SENT
from ._common import CONN_TIME_WAIT
from ._common import FREEBSD  # NOQA
from ._common import LINUX
from ._common import MACOS
from ._common import NETBSD  # NOQA
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import OPENBSD  # NOQA
from ._common import OSX  # deprecated alias
from ._common import POSIX  # NOQA
from ._common import POWER_TIME_UNKNOWN
from ._common import POWER_TIME_UNLIMITED
from ._common import STATUS_DEAD
from ._common import STATUS_DISK_SLEEP
from ._common import STATUS_IDLE
from ._common import STATUS_LOCKED
from ._common import STATUS_PARKED
from ._common import STATUS_RUNNING
from ._common import STATUS_SLEEPING
from ._common import STATUS_STOPPED
from ._common import STATUS_TRACING_STOP
from ._common import STATUS_WAITING
from ._common import STATUS_WAKING
from ._common import STATUS_ZOMBIE
from ._common import SUNOS
from ._common import WINDOWS
from ._common import AccessDenied
from ._common import Error
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import ZombieProcess
from ._common import memoize_when_activated
from ._common import wrap_numbers as _wrap_numbers
from ._compat import PY3 as _PY3
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import SubprocessTimeoutExpired as _SubprocessTimeoutExpired
from ._compat import long
def check_gone(proc, timeout):
    try:
        returncode = proc.wait(timeout=timeout)
    except TimeoutExpired:
        pass
    except _SubprocessTimeoutExpired:
        pass
    else:
        if returncode is not None or not proc.is_running():
            proc.returncode = returncode
            gone.add(proc)
            if callback is not None:
                callback(proc)