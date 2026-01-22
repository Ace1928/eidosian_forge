from __future__ import annotations
import logging
import sys
import typing as t
from werkzeug.local import LocalProxy
from .globals import request
def has_level_handler(logger: logging.Logger) -> bool:
    """Check if there is a handler in the logging chain that will handle the
    given logger's :meth:`effective level <~logging.Logger.getEffectiveLevel>`.
    """
    level = logger.getEffectiveLevel()
    current = logger
    while current:
        if any((handler.level <= level for handler in current.handlers)):
            return True
        if not current.propagate:
            break
        current = current.parent
    return False