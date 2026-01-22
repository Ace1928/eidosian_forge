from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@property
def is_hyb_mgga_family(self) -> bool:
    """True if this functional belongs to the hybrid + meta-GGA family."""
    return self.family == 'HYB_MGGA'