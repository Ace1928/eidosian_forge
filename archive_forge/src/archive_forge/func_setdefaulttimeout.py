import errno
import select
import socket
import six
import sys
from ._exceptions import *
from ._ssl_compat import *
from ._utils import *
def setdefaulttimeout(timeout):
    """
    Set the global timeout setting to connect.

    timeout: default socket timeout time. This value is second.
    """
    global _default_timeout
    _default_timeout = timeout