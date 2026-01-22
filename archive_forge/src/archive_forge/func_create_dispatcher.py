import inspect
import select
import sys
import threading
import time
import traceback
import six
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import *
from . import _logging
def create_dispatcher(self, ping_timeout):
    timeout = ping_timeout or 10
    if self.sock.is_ssl():
        return SSLDispacther(self, timeout)
    return Dispatcher(self, timeout)