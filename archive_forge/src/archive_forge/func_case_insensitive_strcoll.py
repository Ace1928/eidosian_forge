import locale
import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from urllib.parse import urljoin, urlsplit
from .exceptions import xpath_error
def case_insensitive_strcoll(s1: str, s2: str) -> int:
    if s1.casefold() == s2.casefold():
        return 0
    elif s1.casefold() < s2.casefold():
        return -1
    else:
        return 1