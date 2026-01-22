import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds_with_names_single():
    os.environ['LISTEN_FDS'] = '1'
    os.environ['LISTEN_PID'] = str(os.getpid())
    os.environ['LISTEN_FDNAMES'] = 'cmds'
    assert listen_fds_with_names(False) == {3: 'cmds'}
    assert listen_fds_with_names() == {3: 'cmds'}
    assert listen_fds_with_names(True) == {}