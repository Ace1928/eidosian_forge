import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .helpers import (
from .http import (
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
@reify
def links(self) -> 'MultiDictProxy[MultiDictProxy[Union[str, URL]]]':
    links_str = ', '.join(self.headers.getall('link', []))
    if not links_str:
        return MultiDictProxy(MultiDict())
    links: MultiDict[MultiDictProxy[Union[str, URL]]] = MultiDict()
    for val in re.split(',(?=\\s*<)', links_str):
        match = re.match('\\s*<(.*)>(.*)', val)
        if match is None:
            continue
        url, params_str = match.groups()
        params = params_str.split(';')[1:]
        link: MultiDict[Union[str, URL]] = MultiDict()
        for param in params:
            match = re.match('^\\s*(\\S*)\\s*=\\s*([\'\\"]?)(.*?)(\\2)\\s*$', param, re.M)
            if match is None:
                continue
            key, _, value, _ = match.groups()
            link.add(key, value)
        key = link.get('rel', url)
        link.add('url', self.url.join(URL(url)))
        links.add(str(key), MultiDictProxy(link))
    return MultiDictProxy(links)