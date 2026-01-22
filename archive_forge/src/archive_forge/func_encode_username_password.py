import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
def encode_username_password(username: Union[str, bytes], password: Union[str, bytes]) -> bytes:
    """Encodes a username/password pair in the format used by HTTP auth.

    The return value is a byte string in the form ``username:password``.

    .. versionadded:: 5.1
    """
    if isinstance(username, unicode_type):
        username = unicodedata.normalize('NFC', username)
    if isinstance(password, unicode_type):
        password = unicodedata.normalize('NFC', password)
    return utf8(username) + b':' + utf8(password)