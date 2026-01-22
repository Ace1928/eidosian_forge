from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def sorted_linked_pids(self) -> List[int]:
    """
        Returns the sorted linked pids
        """
    if not self.linked_pids:
        return []
    pids = list(self.linked_pids)
    if not self.is_parent:
        pids.append(self.pid)
    if self.parent_pid and self.parent_pid in pids:
        pids.remove(self.parent_pid)
    return sorted(pids, key=int)