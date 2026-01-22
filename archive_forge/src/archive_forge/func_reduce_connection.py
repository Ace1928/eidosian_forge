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
def reduce_connection(conn):
    df = reduction.DupFd(conn.fileno())
    return (rebuild_connection, (df, conn.readable, conn.writable))