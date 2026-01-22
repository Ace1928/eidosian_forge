from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def modified_count(self) -> int:
    """The number of documents modified."""
    self._raise_if_unacknowledged('modified_count')
    return cast(int, self.__bulk_api_result.get('nModified'))