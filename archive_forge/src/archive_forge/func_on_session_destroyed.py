from __future__ import annotations
import logging # isort:skip
import sys
from os.path import (
from types import ModuleType
from typing import TYPE_CHECKING, Any, Coroutine
from jinja2 import Environment, FileSystemLoader, Template
from ...core.types import PathLike
from ...document import Document
from ..application import ServerContext, SessionContext
from .code_runner import CodeRunner
from .handler import Handler
from .notebook import NotebookHandler
from .script import ScriptHandler
from .server_lifecycle import ServerLifecycleHandler
from .server_request_handler import ServerRequestHandler
def on_session_destroyed(self, session_context: SessionContext) -> Coroutine[Any, Any, None]:
    """ Execute ``on_session_destroyed`` from ``server_lifecycle.py`` (if
        it is defined) when a session is destroyed.

        Args:
            session_context (SessionContext) :

        """
    return self._lifecycle_handler.on_session_destroyed(session_context)