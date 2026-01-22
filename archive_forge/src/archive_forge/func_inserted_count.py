from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def inserted_count(self) -> int:
    """The number of documents inserted."""
    self._raise_if_unacknowledged('inserted_count')
    return cast(int, self.__bulk_api_result.get('nInserted'))