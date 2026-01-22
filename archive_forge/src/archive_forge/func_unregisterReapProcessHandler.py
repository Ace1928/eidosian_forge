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
def unregisterReapProcessHandler(pid, process):
    """
    Unregister a process handler previously registered with
    L{registerReapProcessHandler}.
    """
    if not (pid in reapProcessHandlers and reapProcessHandlers[pid] == process):
        raise RuntimeError('Try to unregister a process not registered.')
    del reapProcessHandlers[pid]