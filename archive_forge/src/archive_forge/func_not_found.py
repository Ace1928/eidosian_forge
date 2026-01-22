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
def not_found(self, message: str) -> bytes:
    """Begin a HTTP 404 response and return the text of a message."""
    self._cache_headers = []
    logger.info('Not found: %s', message)
    self.respond(HTTP_NOT_FOUND, 'text/plain')
    return message.encode('ascii')