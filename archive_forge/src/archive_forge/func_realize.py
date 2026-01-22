import functools
from typing import Optional
from .base import VariableTracker
def realize(self) -> VariableTracker:
    """Force construction of the real VariableTracker"""
    if self._cache.vt is None:
        self._cache.realize(self.parents_tracker)
    return self._cache.vt