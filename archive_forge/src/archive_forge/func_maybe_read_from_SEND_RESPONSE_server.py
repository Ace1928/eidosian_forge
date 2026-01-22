import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union
from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate
def maybe_read_from_SEND_RESPONSE_server(buf: ReceiveBuffer) -> Union[InformationalResponse, Response, None]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError('illegal request line')
        return None
    if not lines:
        raise LocalProtocolError('no response line received')
    matches = validate(status_line_re, lines[0], 'illegal status line: {!r}', lines[0])
    http_version = b'1.1' if matches['http_version'] is None else matches['http_version']
    reason = b'' if matches['reason'] is None else matches['reason']
    status_code = int(matches['status_code'])
    class_: Union[Type[InformationalResponse], Type[Response]] = InformationalResponse if status_code < 200 else Response
    return class_(headers=list(_decode_header_lines(lines[1:])), _parsed=True, status_code=status_code, reason=reason, http_version=http_version)