from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def upserted_count(self) -> int:
    """The number of documents upserted."""
    self._raise_if_unacknowledged('upserted_count')
    return cast(int, self.__bulk_api_result.get('nUpserted'))