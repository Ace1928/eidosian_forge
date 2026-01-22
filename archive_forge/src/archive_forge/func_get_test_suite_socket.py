import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def get_test_suite_socket():
    global GLANCE_TEST_SOCKET_FD_STR
    if GLANCE_TEST_SOCKET_FD_STR in os.environ:
        fd = int(os.environ[GLANCE_TEST_SOCKET_FD_STR])
        sock = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
        sock.listen(CONF.backlog)
        del os.environ[GLANCE_TEST_SOCKET_FD_STR]
        os.close(fd)
        return sock
    return None