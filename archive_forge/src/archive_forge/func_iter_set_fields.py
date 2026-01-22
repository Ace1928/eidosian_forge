from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
def iter_set_fields(self) -> Iterator[tuple[str, str]]:
    """
        Return an iterator of (field, value) pairs of none None values
        """
    return ((k, v) for k, v in self.iterfields() if v is not None)