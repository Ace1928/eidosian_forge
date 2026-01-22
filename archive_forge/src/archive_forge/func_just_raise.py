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
def just_raise(*a, **kw):
    if throw_args:
        raise throw_args[1].with_traceback(throw_args[2])
    else:
        raise greenlet.GreenletExit()