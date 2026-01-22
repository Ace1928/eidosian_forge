from __future__ import annotations
import ipaddress
import re
import typing
import idna
from ._exceptions import InvalidURL
def percent_encoded(string: str, safe: str='/') -> str:
    """
    Use percent-encoding to quote a string.
    """
    if is_safe(string, safe=safe):
        return string
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe
    return ''.join([char if char in NON_ESCAPED_CHARS else percent_encode(char) for char in string])