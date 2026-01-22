import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import (
from types import TracebackType
def get_fd_error(self) -> Optional[Exception]:
    errno = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    return socket.error(errno, os.strerror(errno))