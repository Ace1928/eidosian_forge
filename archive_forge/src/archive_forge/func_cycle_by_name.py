from __future__ import annotations
from itertools import count
from .imports import symbol_by_name
def cycle_by_name(name):
    """Get cycle class by name."""
    return symbol_by_name(name, CYCLE_ALIASES)