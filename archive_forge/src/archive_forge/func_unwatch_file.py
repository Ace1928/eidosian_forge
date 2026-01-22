from __future__ import annotations
import errno
import math
import select as __select__
import sys
from numbers import Integral
from . import fileno
from .compat import detect_environment
def unwatch_file(self, fd):
    ev = kevent(fd, filter=KQ_FILTER_VNODE, flags=KQ_EV_DELETE, fflags=self.w_fflags)
    self._kcontrol([ev], 0)