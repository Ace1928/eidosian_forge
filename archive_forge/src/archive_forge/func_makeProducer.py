from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def makeProducer(self, request, fileForReading):
    """
        Make a L{StaticProducer} that will produce the body of this response.

        This method will also set the response code and Content-* headers.

        @param request: The L{twisted.web.http.Request} object.
        @param fileForReading: The file object containing the resource.
        @return: A L{StaticProducer}.  Calling C{.start()} on this will begin
            producing the response.
        """
    byteRange = request.getHeader(b'range')
    if byteRange is None:
        self._setContentHeaders(request)
        request.setResponseCode(http.OK)
        return NoRangeStaticProducer(request, fileForReading)
    try:
        parsedRanges = self._parseRangeHeader(byteRange)
    except ValueError:
        log.msg(f'Ignoring malformed Range header {byteRange.decode()!r}')
        self._setContentHeaders(request)
        request.setResponseCode(http.OK)
        return NoRangeStaticProducer(request, fileForReading)
    if len(parsedRanges) == 1:
        offset, size = self._doSingleRangeRequest(request, parsedRanges[0])
        self._setContentHeaders(request, size)
        return SingleRangeStaticProducer(request, fileForReading, offset, size)
    else:
        rangeInfo = self._doMultipleRangeRequest(request, parsedRanges)
        return MultipleRangeStaticProducer(request, fileForReading, rangeInfo)