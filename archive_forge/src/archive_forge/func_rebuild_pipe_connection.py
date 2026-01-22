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
def rebuild_pipe_connection(dh, readable, writable):
    handle = dh.detach()
    return PipeConnection(handle, readable, writable)