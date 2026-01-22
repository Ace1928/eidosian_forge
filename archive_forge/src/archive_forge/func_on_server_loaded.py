from __future__ import annotations
import logging # isort:skip
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any
from ...document import Document
from ..application import ServerContext, SessionContext
def on_server_loaded(self, server_context: ServerContext) -> None:
    """ Execute code when the server is first started.

        Subclasses may implement this method to provide for any one-time
        initialization that is necessary after the server starts, but
        before any sessions are created.

        Args:
            server_context (ServerContext) :

        """
    pass