from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def job_reference_scoped(self) -> str:
    assert self.entity
    assert self.project
    unscoped = self.job_reference()
    return f'{self.entity}/{self.project}/{unscoped}'