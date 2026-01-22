from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def is_job_or_run(self) -> bool:
    return self.is_job() or self.is_run()