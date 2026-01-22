import locale
import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from urllib.parse import urljoin, urlsplit
from .exceptions import xpath_error
def case_insensitive_strxfrm(s: str) -> str:
    return s.casefold()