from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
@property
def server_context(self) -> ServerContext:
    """ The server context for this session context

        """
    return self._server_context