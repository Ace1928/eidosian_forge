import io
import os
import msvcrt
import signal
import sys
from . import context
from . import spawn
from . import reduction
from .compat import _winapi
def close_thread_handle(handle):
    handle.Close()