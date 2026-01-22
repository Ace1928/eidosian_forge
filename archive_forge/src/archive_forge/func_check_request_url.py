import logging
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from h2.errors import ErrorCodes
from h2.exceptions import H2Error, ProtocolError, StreamClosedError
from hpack import HeaderTuple
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.error import ConnectionClosed
from twisted.python.failure import Failure
from twisted.web.client import ResponseFailed
from scrapy.http import Request
from scrapy.http.headers import Headers
from scrapy.responsetypes import responsetypes
def check_request_url(self) -> bool:
    url = urlparse(self._request.url)
    return url.netloc == str(self._protocol.metadata['uri'].host, 'utf-8') or url.netloc == str(self._protocol.metadata['uri'].netloc, 'utf-8') or url.netloc == f'{self._protocol.metadata['ip_address']}:{self._protocol.metadata['uri'].port}'