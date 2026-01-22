from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def has_unstable_default(self, obj: HasProps) -> bool:
    return self.property._may_have_unstable_default() or self.is_unstable(obj.__overridden_defaults__.get(self.name, None))