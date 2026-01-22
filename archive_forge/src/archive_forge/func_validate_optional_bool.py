from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
@staticmethod
def validate_optional_bool(v: Any) -> bool:
    return v is None or isinstance(v, bool)