import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds():
    os.environ['LISTEN_FDS'] = '3'
    os.environ['LISTEN_PID'] = str(os.getpid())
    assert listen_fds(False) == [3, 4, 5]
    assert listen_fds(True) == [3, 4, 5]
    assert listen_fds() == []