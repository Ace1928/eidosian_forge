from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def parent_cmdline(self) -> Optional[List[str]]:
    """
        Parent Command Line
        """
    return self.parent.cmdline() if self.parent else None