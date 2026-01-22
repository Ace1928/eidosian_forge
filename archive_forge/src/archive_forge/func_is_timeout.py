import functools
import inspect
import eventlet
from eventlet.support import greenlets as greenlet
from eventlet.hubs import get_hub
@property
def is_timeout(self):
    return True