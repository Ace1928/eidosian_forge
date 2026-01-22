from __future__ import annotations
import logging # isort:skip
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any
from ...document import Document
from ..application import ServerContext, SessionContext
def static_path(self) -> str | None:
    """ Return a path to app-specific static resources, if applicable.

        """
    if self.failed:
        return None
    else:
        return self._static