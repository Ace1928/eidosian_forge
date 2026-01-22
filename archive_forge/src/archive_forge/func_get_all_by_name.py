from __future__ import annotations
import logging # isort:skip
import contextlib
import weakref
from typing import (
from ..core.types import ID
from ..model import Model
from ..util.datatypes import MultiValuedDict
def get_all_by_name(self, name: str) -> list[Model]:
    """ Find all the models for this Document with a given name.

        Args:
            name (str) : the name of a model to search for

        Returns
            A list of models

        """
    return self._models_by_name.get_all(name)