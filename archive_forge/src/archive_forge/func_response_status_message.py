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
def response_status_message(status: Union[bytes, float, int, str]) -> str:
    """Return status code plus status text descriptive message"""
    status_int = int(status)
    message = http.RESPONSES.get(status_int, 'Unknown Status')
    return f'{status_int} {to_unicode(message)}'