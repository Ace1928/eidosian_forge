import errno
import fcntl
import os
import eventlet
import eventlet.debug
import eventlet.greenthread
import eventlet.hubs
Close the mutex.

        This releases its file descriptors.
        You can't use a mutex after it's been closed.
        