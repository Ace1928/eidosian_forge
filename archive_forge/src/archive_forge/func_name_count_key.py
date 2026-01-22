from __future__ import annotations
import contextlib
from abc import ABC
from typing import (
@property
def name_count_key(self) -> str:
    """
        Returns the count key
        """
    return f'{self.name_prefix}.{self.kdb_type}.{self.name}:count'