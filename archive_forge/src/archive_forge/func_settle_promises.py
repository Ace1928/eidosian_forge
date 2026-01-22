from collections import deque
from threading import local
def settle_promises(self, promise):
    if self.trampoline_enabled:
        self._async_settle_promise(promise)
    else:
        promise.scheduler.call(promise._settle_promises)