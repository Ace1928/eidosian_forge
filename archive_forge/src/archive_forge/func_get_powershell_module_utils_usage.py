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
def get_powershell_module_utils_usage(self, path: str) -> list[str]:
    """Return a list of paths which depend on the given path which is a PowerShell module_utils file."""
    if not self.powershell_module_utils_imports:
        display.info('Analyzing powershell module_utils imports...')
        before = time.time()
        self.powershell_module_utils_imports = get_powershell_module_utils_imports(self.powershell_targets)
        after = time.time()
        display.info('Processed %d powershell module_utils in %d second(s).' % (len(self.powershell_module_utils_imports), after - before))
    name = get_powershell_module_utils_name(path)
    return sorted(self.powershell_module_utils_imports[name])