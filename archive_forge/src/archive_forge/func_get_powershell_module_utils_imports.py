from __future__ import annotations
import os
import re
from ..io import (
from ..util import (
from .common import (
from ..data import (
from ..target import (
def get_powershell_module_utils_imports(powershell_targets: list[TestTarget]) -> dict[str, set[str]]:
    """Return a dictionary of module_utils names mapped to sets of powershell file paths."""
    module_utils = enumerate_module_utils()
    imports_by_target_path = {}
    for target in powershell_targets:
        imports_by_target_path[target.path] = extract_powershell_module_utils_imports(target.path, module_utils)
    imports: dict[str, set[str]] = {module_util: set() for module_util in module_utils}
    for target_path, modules in imports_by_target_path.items():
        for module_util in modules:
            imports[module_util].add(target_path)
    for module_util in sorted(imports):
        if not imports[module_util]:
            display.warning('No imports found which use the "%s" module_util.' % module_util)
    return imports