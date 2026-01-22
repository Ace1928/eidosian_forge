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
@classmethod
def get_content_version(cls, abspath: str) -> str:
    """Returns a version string for the resource at the given path.

        This class method may be overridden by subclasses.  The
        default implementation is a SHA-512 hash of the file's contents.

        .. versionadded:: 3.1
        """
    data = cls.get_content(abspath)
    hasher = hashlib.sha512()
    if isinstance(data, bytes):
        hasher.update(data)
    else:
        for chunk in data:
            hasher.update(chunk)
    return hasher.hexdigest()