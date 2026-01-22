from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from ..core.types import ID
@property
def reqid(self) -> ID:
    """ The request ID of the originating message. """
    return self._reqid