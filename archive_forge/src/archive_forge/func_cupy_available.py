from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
def cupy_available():
    return _CUPY_AVAILABLE