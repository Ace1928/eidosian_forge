import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import (
from mypy_extensions import mypyc_attr
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from black.handle_ipynb_magics import jupyter_dependencies_are_installed
from black.mode import TargetVersion
from black.output import err
from black.report import Report
def best_effort_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.absolute().relative_to(root)
    except ValueError:
        pass
    root_parent = next((p for p in path.parents if _cached_resolve(p) == root), None)
    if root_parent is not None:
        return path.relative_to(root_parent)
    return _cached_resolve(path).relative_to(root)