from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def is_run(self) -> bool:
    return self.ref_type == ReferenceType.RUN