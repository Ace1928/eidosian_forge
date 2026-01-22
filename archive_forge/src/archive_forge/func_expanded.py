from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
@property
def expanded(self) -> bool:
    """True if the expander was created with `expanded=True`."""
    return self._expanded