from __future__ import annotations
import inspect
import signal
import sys
from functools import wraps
from typing import TYPE_CHECKING, Final, Protocol, TypeVar
import attrs
from .._util import is_main_thread
def currently_ki_protected() -> bool:
    """Check whether the calling code has :exc:`KeyboardInterrupt` protection
    enabled.

    It's surprisingly easy to think that one's :exc:`KeyboardInterrupt`
    protection is enabled when it isn't, or vice-versa. This function tells
    you what Trio thinks of the matter, which makes it useful for ``assert``\\s
    and unit tests.

    Returns:
      bool: True if protection is enabled, and False otherwise.

    """
    return ki_protection_enabled(sys._getframe())