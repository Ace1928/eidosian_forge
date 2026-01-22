import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def makeGreenFile(self, *args, **kw):
    warnings.warn('makeGreenFile has been deprecated, please use makefile instead', DeprecationWarning, stacklevel=2)
    return self.makefile(*args, **kw)