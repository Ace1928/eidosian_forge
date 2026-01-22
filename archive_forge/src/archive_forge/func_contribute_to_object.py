import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def contribute_to_object(self, obj):
    obj.inq, obj.outq, obj.synq = (self.inq, self.outq, self.synq)
    obj.inqW_fd = self.inq._writer.fileno()
    obj.outqR_fd = self.outq._reader.fileno()
    if self.synq:
        obj.synqR_fd = self.synq._reader.fileno()
        obj.synqW_fd = self.synq._writer.fileno()
        obj.send_syn_offset = _get_send_offset(self.synq._writer)
    else:
        obj.synqR_fd = obj.synqW_fd = obj._send_syn_offset = None
    obj._quick_put = self.inq._writer.send
    obj._quick_get = self.outq._reader.recv
    obj.send_job_offset = _get_send_offset(self.inq._writer)
    return obj