import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
def join_thread(self):
    debug('Queue.join_thread()')
    assert self._closed, 'Queue {0!r} not closed'.format(self)
    if self._jointhread:
        self._jointhread()