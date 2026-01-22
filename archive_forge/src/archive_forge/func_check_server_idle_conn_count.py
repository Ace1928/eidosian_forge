import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
def check_server_idle_conn_count(count, timeout=1.0):
    deadline = time.time() + timeout
    while True:
        n = test_client.server_instance._connections._num_connections
        if n == count:
            return
        assert time.time() <= deadline, ('idle conn count mismatch, wanted {count}, got {n}'.format(**locals()),)