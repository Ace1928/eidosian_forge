from __future__ import annotations
import collections
import os
import re
import time
import typing as t
from ..target import (
from ..util import (
from .python import (
from .csharp import (
from .powershell import (
from ..config import (
from ..metadata import (
from ..data import (
def get_csharp_module_utils_usage(self, path: str) -> list[str]:
    """Return a list of paths which depend on the given path which is a C# module_utils file."""
    if not self.csharp_module_utils_imports:
        display.info('Analyzing C# module_utils imports...')
        before = time.time()
        self.csharp_module_utils_imports = get_csharp_module_utils_imports(self.powershell_targets, self.csharp_targets)
        after = time.time()
        display.info('Processed %d C# module_utils in %d second(s).' % (len(self.csharp_module_utils_imports), after - before))
    name = get_csharp_module_utils_name(path)
    return sorted(self.csharp_module_utils_imports[name])