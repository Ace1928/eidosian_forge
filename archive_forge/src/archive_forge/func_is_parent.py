from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def is_parent(self) -> bool:
    """
        If the parent process is none, we assume it is the current
        process. This is the case when running in a single worker
        """
    return self.pid == self.parent_pid if self.parent_pid is not None else True