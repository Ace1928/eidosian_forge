import asyncio
import base64
import binascii
import hashlib
import json
import sys
from typing import Any, Final, Iterable, Optional, Tuple, cast
import attr
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import call_later, set_result
from .http import (
from .log import ws_logger
from .streams import EofStream, FlowControlDataQueue
from .typedefs import JSONDecoder, JSONEncoder
from .web_exceptions import HTTPBadRequest, HTTPException
from .web_request import BaseRequest
from .web_response import StreamResponse
def can_prepare(self, request: BaseRequest) -> WebSocketReady:
    if self._writer is not None:
        raise RuntimeError('Already started')
    try:
        _, protocol, _, _ = self._handshake(request)
    except HTTPException:
        return WebSocketReady(False, None)
    else:
        return WebSocketReady(True, protocol)