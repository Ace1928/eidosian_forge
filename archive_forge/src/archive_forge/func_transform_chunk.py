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
def transform_chunk(self, chunk: bytes, finishing: bool) -> bytes:
    if self._gzipping:
        self._gzip_file.write(chunk)
        if finishing:
            self._gzip_file.close()
        else:
            self._gzip_file.flush()
        chunk = self._gzip_value.getvalue()
        self._gzip_value.truncate(0)
        self._gzip_value.seek(0)
    return chunk