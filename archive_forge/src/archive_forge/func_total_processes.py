from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def total_processes(self) -> int:
    """
        Returns the total number of processes
        """
    return len(self.linked_pids) + 1 if self.linked_pids else 1