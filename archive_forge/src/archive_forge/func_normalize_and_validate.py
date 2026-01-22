import re
from typing import AnyStr, cast, List, overload, Sequence, Tuple, TYPE_CHECKING, Union
from ._abnf import field_name, field_value
from ._util import bytesify, LocalProtocolError, validate
def normalize_and_validate(headers: Union[Headers, HeaderTypes], _parsed: bool=False) -> Headers:
    new_headers = []
    seen_content_length = None
    saw_transfer_encoding = False
    for name, value in headers:
        if not _parsed:
            name = bytesify(name)
            value = bytesify(value)
            validate(_field_name_re, name, 'Illegal header name {!r}', name)
            validate(_field_value_re, value, 'Illegal header value {!r}', value)
        assert isinstance(name, bytes)
        assert isinstance(value, bytes)
        raw_name = name
        name = name.lower()
        if name == b'content-length':
            lengths = {length.strip() for length in value.split(b',')}
            if len(lengths) != 1:
                raise LocalProtocolError('conflicting Content-Length headers')
            value = lengths.pop()
            validate(_content_length_re, value, 'bad Content-Length')
            if seen_content_length is None:
                seen_content_length = value
                new_headers.append((raw_name, name, value))
            elif seen_content_length != value:
                raise LocalProtocolError('conflicting Content-Length headers')
        elif name == b'transfer-encoding':
            if saw_transfer_encoding:
                raise LocalProtocolError('multiple Transfer-Encoding headers', error_status_hint=501)
            value = value.lower()
            if value != b'chunked':
                raise LocalProtocolError('Only Transfer-Encoding: chunked is supported', error_status_hint=501)
            saw_transfer_encoding = True
            new_headers.append((raw_name, name, value))
        else:
            new_headers.append((raw_name, name, value))
    return Headers(new_headers)