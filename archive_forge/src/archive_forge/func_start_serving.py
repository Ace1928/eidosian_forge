import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def start_serving(self, delegate: httputil.HTTPServerConnectionDelegate) -> None:
    """Starts serving requests on this connection.

        :arg delegate: a `.HTTPServerConnectionDelegate`
        """
    assert isinstance(delegate, httputil.HTTPServerConnectionDelegate)
    fut = gen.convert_yielded(self._server_request_loop(delegate))
    self._serving_future = fut
    self.stream.io_loop.add_future(fut, lambda f: f.result())