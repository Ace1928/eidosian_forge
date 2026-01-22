from __future__ import annotations
from typing import Any
from .schema import EventSchema
@property
def schema_ids(self) -> list[str]:
    return list(self._schemas.keys())