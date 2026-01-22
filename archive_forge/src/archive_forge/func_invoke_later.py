from collections import deque
from threading import local
def invoke_later(self, fn):
    if self.trampoline_enabled:
        self._async_invoke_later(fn, scheduler)
    else:
        scheduler.call_later(0.1, fn)