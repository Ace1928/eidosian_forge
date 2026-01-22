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
def resolves_outside_root_or_cannot_stat(path: Path, root: Path, report: Optional[Report]=None) -> bool:
    """
    Returns whether the path is a symbolic link that points outside the
    root directory. Also returns True if we failed to resolve the path.
    """
    try:
        if sys.version_info < (3, 8, 6):
            path = path.absolute()
        resolved_path = _cached_resolve(path)
    except OSError as e:
        if report:
            report.path_ignored(path, f'cannot be read because {e}')
        return True
    try:
        resolved_path.relative_to(root)
    except ValueError:
        if report:
            report.path_ignored(path, f'is a symbolic link that points outside {root}')
        return True
    return False