from eventlet import hubs
from eventlet.support import greenlets as greenlet
def has_exception(self):
    return self._exc is not None