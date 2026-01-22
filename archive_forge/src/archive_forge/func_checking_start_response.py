from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
def checking_start_response(*args: t.Any, **kwargs: t.Any) -> t.Callable[[bytes], None]:
    if len(args) not in {2, 3}:
        warn(f'Invalid number of arguments: {len(args)}, expected 2 or 3.', WSGIWarning, stacklevel=2)
    if kwargs:
        warn("'start_response' does not take keyword arguments.", WSGIWarning)
    status: str = args[0]
    headers: list[tuple[str, str]] = args[1]
    exc_info: None | tuple[type[BaseException], BaseException, TracebackType] = args[2] if len(args) == 3 else None
    headers_set[:] = self.check_start_response(status, headers, exc_info)
    return GuardedWrite(start_response(status, headers, exc_info), chunks)