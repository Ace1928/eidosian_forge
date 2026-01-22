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
def clear_all_cookies(self, **kwargs: Any) -> None:
    """Attempt to delete all the cookies the user sent with this request.

        See `clear_cookie` for more information on keyword arguments. Due to
        limitations of the cookie protocol, it is impossible to determine on the
        server side which values are necessary for the ``domain``, ``path``,
        ``samesite``, or ``secure`` arguments, this method can only be
        successful if you consistently use the same values for these arguments
        when setting cookies.

        Similar to `set_cookie`, the effect of this method will not be seen
        until the following request.

        .. versionchanged:: 3.2

           Added the ``path`` and ``domain`` parameters.

        .. versionchanged:: 6.3

           Now accepts all keyword arguments that ``set_cookie`` does.

        .. deprecated:: 6.3

           The increasingly complex rules governing cookies have made it
           impossible for a ``clear_all_cookies`` method to work reliably
           since all we know about cookies are their names. Applications
           should generally use ``clear_cookie`` one at a time instead.
        """
    for name in self.request.cookies:
        self.clear_cookie(name, **kwargs)