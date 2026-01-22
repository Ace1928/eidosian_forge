import inspect
import sys
from collections import deque
from weakref import WeakMethod, ref
from .abstract import Thenable
from .utils import reraise
def throw1(self, exc=None):
    if not self.cancelled:
        exc = exc if exc is not None else sys.exc_info()[1]
        self.failed, self.reason = (True, exc)
        if self.on_error:
            self.on_error(*self.args + (exc,), **self.kwargs)