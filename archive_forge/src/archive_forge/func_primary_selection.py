from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
@property
def primary_selection(self) -> Selection:
    primaries = [self.primary] if self.primary else []
    return self.with_server_descriptions(primaries)