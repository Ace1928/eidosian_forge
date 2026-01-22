from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
def call_after_local(seconds, function, *args, **kwargs):
    warnings.warn('call_after_local is renamed to spawn_after_local, whichhas the same signature and semantics (plus a bit extra).', DeprecationWarning, stacklevel=2)
    hub = hubs.get_hub()
    g = greenlet.greenlet(function, parent=hub.greenlet)
    t = hub.schedule_call_local(seconds, g.switch, *args, **kwargs)
    return t