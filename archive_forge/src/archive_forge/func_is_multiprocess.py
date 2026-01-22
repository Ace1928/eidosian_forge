from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def is_multiprocess(self) -> bool:
    """
        If the linked pids are empty, we assume it is a single
        process. This is the case when running in a single worker
        """
    return any(('multiprocessing' in cmd for cmd in self.cmdline))