from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
def check_start_response(self, status: str, headers: list[tuple[str, str]], exc_info: None | tuple[type[BaseException], BaseException, TracebackType]) -> tuple[int, Headers]:
    check_type('status', status, str)
    status_code_str = status.split(None, 1)[0]
    if len(status_code_str) != 3 or not status_code_str.isdecimal():
        warn('Status code must be three digits.', WSGIWarning, stacklevel=3)
    if len(status) < 4 or status[3] != ' ':
        warn(f'Invalid value for status {status!r}. Valid status strings are three digits, a space and a status explanation.', WSGIWarning, stacklevel=3)
    status_code = int(status_code_str)
    if status_code < 100:
        warn('Status code < 100 detected.', WSGIWarning, stacklevel=3)
    if type(headers) is not list:
        warn('Header list is not a list.', WSGIWarning, stacklevel=3)
    for item in headers:
        if type(item) is not tuple or len(item) != 2:
            warn('Header items must be 2-item tuples.', WSGIWarning, stacklevel=3)
        name, value = item
        if type(name) is not str or type(value) is not str:
            warn('Header keys and values must be strings.', WSGIWarning, stacklevel=3)
        if name.lower() == 'status':
            warn('The status header is not supported due to conflicts with the CGI spec.', WSGIWarning, stacklevel=3)
    if exc_info is not None and (not isinstance(exc_info, tuple)):
        warn('Invalid value for exc_info.', WSGIWarning, stacklevel=3)
    headers = Headers(headers)
    self.check_headers(headers)
    return (status_code, headers)