from typing import (
from urllib.parse import urlparse
def enforce_bytes(value: Union[bytes, str], *, name: str) -> bytes:
    """
    Any arguments that are ultimately represented as bytes can be specified
    either as bytes or as strings.

    However we enforce that any string arguments must only contain characters in
    the plain ASCII range. chr(0)...chr(127). If you need to use characters
    outside that range then be precise, and use a byte-wise argument.
    """
    if isinstance(value, str):
        try:
            return value.encode('ascii')
        except UnicodeEncodeError:
            raise TypeError(f'{name} strings may not include unicode characters.')
    elif isinstance(value, bytes):
        return value
    seen_type = type(value).__name__
    raise TypeError(f'{name} must be bytes or str, but got {seen_type}.')