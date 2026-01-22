import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
def stream_request_body(cls: Type[_RequestHandlerType]) -> Type[_RequestHandlerType]:
    """Apply to `RequestHandler` subclasses to enable streaming body support.

    This decorator implies the following changes:

    * `.HTTPServerRequest.body` is undefined, and body arguments will not
      be included in `RequestHandler.get_argument`.
    * `RequestHandler.prepare` is called when the request headers have been
      read instead of after the entire body has been read.
    * The subclass must define a method ``data_received(self, data):``, which
      will be called zero or more times as data is available.  Note that
      if the request has an empty body, ``data_received`` may not be called.
    * ``prepare`` and ``data_received`` may return Futures (such as via
      ``@gen.coroutine``, in which case the next method will not be called
      until those futures have completed.
    * The regular HTTP method (``post``, ``put``, etc) will be called after
      the entire body has been read.

    See the `file receiver demo <https://github.com/tornadoweb/tornado/tree/stable/demos/file_upload/>`_
    for example usage.
    """
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r', cls)
    cls._stream_request_body = True
    return cls