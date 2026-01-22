import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
Wait for the next result, suspending the current greenthread until it
        is available.  Raises StopIteration when there are no more results.