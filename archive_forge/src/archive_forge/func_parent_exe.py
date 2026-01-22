from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def parent_exe(self) -> Optional[str]:
    """
        Parent Executable
        """
    return self.parent.exe() if self.parent else None