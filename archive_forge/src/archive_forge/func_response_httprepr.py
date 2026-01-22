import os
import re
import tempfile
import webbrowser
from typing import Any, Callable, Iterable, Tuple, Union
from weakref import WeakKeyDictionary
from twisted.web import http
from w3lib import html
import scrapy
from scrapy.http.response import Response
from scrapy.utils.decorators import deprecated
from scrapy.utils.python import to_bytes, to_unicode
@deprecated
def response_httprepr(response: Response) -> bytes:
    """Return raw HTTP representation (as bytes) of the given response. This
    is provided only for reference, since it's not the exact stream of bytes
    that was received (that's not exposed by Twisted).
    """
    values = [b'HTTP/1.1 ', to_bytes(str(response.status)), b' ', to_bytes(http.RESPONSES.get(response.status, b'')), b'\r\n']
    if response.headers:
        values.extend([response.headers.to_string(), b'\r\n'])
    values.extend([b'\r\n', response.body])
    return b''.join(values)