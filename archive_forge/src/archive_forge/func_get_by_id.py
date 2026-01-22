from __future__ import annotations
import logging # isort:skip
import contextlib
import weakref
from typing import (
from ..core.types import ID
from ..model import Model
from ..util.datatypes import MultiValuedDict
def get_by_id(self, id: ID) -> Model | None:
    """ Find the model for this Document with a given ID.

        Args:
            id (ID) : model ID to search for
                If no model with the given ID exists, returns None

        Return:
            a Model or None

        """
    return self._models.get(id, None)