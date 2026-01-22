from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def recurse_import(import_name: str, depth: int=0, seen: t.Optional[set[str]]=None) -> set[str]:
    """Recursively expand module_utils imports from module_utils files."""
    display.info('module_utils import: %s%s' % ('  ' * depth, import_name), verbosity=4)
    if seen is None:
        seen = {import_name}
    results = {import_name}
    if import_name in VIRTUAL_PACKAGES:
        for sub_import in sorted(virtual_utils):
            if sub_import.startswith('%s.' % import_name):
                if sub_import in seen:
                    continue
                seen.add(sub_import)
                matches = sorted(recurse_import(sub_import, depth + 1, seen))
                for result in matches:
                    results.add(result)
    import_path = get_import_path(import_name)
    if import_path not in imports_by_target_path:
        import_path = get_import_path(import_name, package=True)
        if import_path not in imports_by_target_path:
            raise ApplicationError('Cannot determine path for module_utils import: %s' % import_name)
    for name in sorted(imports_by_target_path[import_path], reverse=True):
        if name in virtual_utils:
            continue
        if name in seen:
            continue
        seen.add(name)
        matches = sorted(recurse_import(name, depth + 1, seen))
        for result in matches:
            results.add(result)
    return results