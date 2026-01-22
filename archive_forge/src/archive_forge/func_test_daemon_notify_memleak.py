import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_daemon_notify_memleak():
    fd = 1
    fds = [fd]
    ref_cnt = sys.getrefcount(fd)
    try:
        notify('', True, 0, fds)
    except connection_error:
        pass
    assert sys.getrefcount(fd) <= ref_cnt, 'leak'