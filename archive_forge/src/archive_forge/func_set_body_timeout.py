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
def set_body_timeout(self, timeout: float) -> None:
    """Sets the body timeout for a single request.

        Overrides the value from `.HTTP1ConnectionParameters`.
        """
    self._body_timeout = timeout