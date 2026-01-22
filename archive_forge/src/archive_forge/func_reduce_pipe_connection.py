import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def reduce_pipe_connection(conn):
    access = (_winapi.FILE_GENERIC_READ if conn.readable else 0) | (_winapi.FILE_GENERIC_WRITE if conn.writable else 0)
    dh = reduction.DupHandle(conn.fileno(), access)
    return (rebuild_pipe_connection, (dh, conn.readable, conn.writable))