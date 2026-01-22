from __future__ import annotations
from .base import *
def setcount(self, value: Numeric, ex: Optional[int]=None) -> Numeric:
    """
        Sets the count for the given key
        """
    self.kdb.set(self.name_count_key, value, ex=ex or self.expiration, _serializer=True)
    return value