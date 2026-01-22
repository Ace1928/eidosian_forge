import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
def respond(self, status: str=HTTP_OK, content_type: Optional[str]=None, headers: Optional[List[Tuple[str, str]]]=None):
    """Begin a response with the given status and other headers."""
    if headers:
        self._headers.extend(headers)
    if content_type:
        self._headers.append(('Content-Type', content_type))
    self._headers.extend(self._cache_headers)
    return self._start_response(status, self._headers)