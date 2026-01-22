import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
from . import connection
from . import process
from . import reduction
from . import semaphore_tracker
from . import spawn
from . import util
from .compat import spawnv_passfds
def read_unsigned(fd):
    data = b''
    length = UNSIGNED_STRUCT.size
    while len(data) < length:
        s = os.read(fd, length - len(data))
        if not s:
            raise EOFError('unexpected EOF')
        data += s
    return UNSIGNED_STRUCT.unpack(data)[0]