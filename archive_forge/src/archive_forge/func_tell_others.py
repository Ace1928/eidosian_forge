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
def tell_others(self):
    outqueue = self.outqueue
    put = self.put
    pool = self.pool
    try:
        debug('task handler sending sentinel to result handler')
        outqueue.put(None)
        debug('task handler sending sentinel to workers')
        for p in pool:
            put(None)
    except IOError:
        debug('task handler got IOError when sending sentinels')
    debug('task handler exiting')