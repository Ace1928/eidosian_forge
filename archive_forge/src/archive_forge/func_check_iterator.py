from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
def check_iterator(self, app_iter: t.Iterable[bytes]) -> None:
    if isinstance(app_iter, str):
        warn('The application returned a string. The response will send one character at a time to the client, which will kill performance. Return a list or iterable instead.', WSGIWarning, stacklevel=3)