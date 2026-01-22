from __future__ import annotations
from .base import *
@property
def name_lookup_key(self) -> str:
    """
        Returns the lookup key
        """
    return f'{self.name_prefix}.{self.kdb_type}.{self.name}:{self.primary_key}:lookup'