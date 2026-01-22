import contextlib
import errno
import functools
import os
import signal
import sys
import time
from collections import namedtuple
from . import _common
from ._common import ENCODING
from ._common import ENCODING_ERRS
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import debug
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import parse_environ_block
from ._common import usage_percent
from ._compat import PY3
from ._compat import long
from ._compat import lru_cache
from ._compat import range
from ._compat import unicode
from ._psutil_windows import ABOVE_NORMAL_PRIORITY_CLASS
from ._psutil_windows import BELOW_NORMAL_PRIORITY_CLASS
from ._psutil_windows import HIGH_PRIORITY_CLASS
from ._psutil_windows import IDLE_PRIORITY_CLASS
from ._psutil_windows import NORMAL_PRIORITY_CLASS
from ._psutil_windows import REALTIME_PRIORITY_CLASS
def retry_error_partial_copy(fun):
    """Workaround for https://github.com/giampaolo/psutil/issues/875.
    See: https://stackoverflow.com/questions/4457745#4457745.
    """

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        delay = 0.0001
        times = 33
        for _ in range(times):
            try:
                return fun(self, *args, **kwargs)
            except WindowsError as _:
                err = _
                if err.winerror == ERROR_PARTIAL_COPY:
                    time.sleep(delay)
                    delay = min(delay * 2, 0.04)
                    continue
                raise
        msg = "{} retried {} times, converted to AccessDenied as it's stillreturning {}".format(fun, times, err)
        raise AccessDenied(pid=self.pid, name=self._name, msg=msg)
    return wrapper