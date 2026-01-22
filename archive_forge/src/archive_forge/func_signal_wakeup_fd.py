import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
@contextlib.contextmanager
def signal_wakeup_fd(*, warn_on_full_buffer=False):
    r, w = socket.socketpair()
    old_fd = None
    try:
        r.setblocking(False)
        w.setblocking(False)
        old_fd = signal.set_wakeup_fd(w.fileno(), warn_on_full_buffer=warn_on_full_buffer)
        yield r
    finally:
        if old_fd is not None:
            signal.set_wakeup_fd(old_fd)
        r.close()
        w.close()