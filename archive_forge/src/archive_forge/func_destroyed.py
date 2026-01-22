from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
@property
@abstractmethod
def destroyed(self) -> bool:
    """ If ``True``, the session has been discarded and cannot be used.

        A new session with the same ID could be created later but this instance
        will not come back to life.

        """
    pass