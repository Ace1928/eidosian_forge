from __future__ import annotations
import os
from math import ceil
from kombu.utils.objects import cached_property
def load_average() -> tuple[float, ...]:
    """Return system load average as a triple."""
    return _load_average()