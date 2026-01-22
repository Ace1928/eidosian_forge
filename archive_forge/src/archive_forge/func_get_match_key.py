from __future__ import annotations
from .base import *
def get_match_key(self, key: Optional[str]=None) -> str:
    """
        Returns the match key for the given key
        """
    if key is None:
        return self.name_match_key
    key = str(key)
    if '*' not in key:
        key = f'{key}*'
    return f'{self.name_key}:{key}' if self.name_prefix_enabled and self.name_key not in key else key