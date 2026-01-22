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
def safe_apply_callback(self, fun, *args, **kwargs):
    if fun:
        try:
            fun(*args, **kwargs)
        except self._callbacks_propagate:
            raise
        except Exception as exc:
            error('Pool callback raised exception: %r', exc, exc_info=1)