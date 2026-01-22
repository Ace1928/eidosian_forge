from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@property
def is_x_kind(self) -> bool:
    """True if this is an exchange-only functional."""
    return self.kind == 'EXCHANGE'