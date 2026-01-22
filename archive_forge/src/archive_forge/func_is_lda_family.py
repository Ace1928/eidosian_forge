from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@property
def is_lda_family(self) -> bool:
    """True if this functional belongs to the LDA family."""
    return self.family == 'LDA'