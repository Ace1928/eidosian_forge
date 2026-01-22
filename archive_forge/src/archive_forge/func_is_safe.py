from __future__ import annotations
import ipaddress
import re
import typing
import idna
from ._exceptions import InvalidURL
def is_safe(string: str, safe: str='/') -> bool:
    """
    Determine if a given string is already quote-safe.
    """
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe + '%'
    for char in string:
        if char not in NON_ESCAPED_CHARS:
            return False
    return True