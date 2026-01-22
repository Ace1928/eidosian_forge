import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
@contextlib.contextmanager
def skip_enosys():
    try:
        yield
    except OSError as e:
        if e.errno == errno.ENOSYS:
            pytest.skip()
        raise