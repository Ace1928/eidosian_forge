from __future__ import annotations
from .base import *
def getcount(self) -> Numeric:
    """
        Returns the count for the given key
        """
    return int(self.kdb.get(self.name_count_key, 0, _return_raw_value=True, _serializer=True))