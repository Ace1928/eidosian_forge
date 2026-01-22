from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generator
def get_app_or_none() -> Application[Any] | None:
    """
    Get the current active (running) Application, or return `None` if no
    application is running.
    """
    session = _current_app_session.get()
    return session.app