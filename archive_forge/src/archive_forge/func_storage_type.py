from typing import Any
from triad import assert_or_throw
from triad.utils.hash import to_uuid
@property
def storage_type(self) -> str:
    """The storage type of this yield"""
    return self._storage_type