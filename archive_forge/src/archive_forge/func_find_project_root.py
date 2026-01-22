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
@lru_cache
def find_project_root(srcs: Sequence[str], stdin_filename: Optional[str]=None) -> Tuple[Path, str]:
    """Return a directory containing .git, .hg, or pyproject.toml.

    That directory will be a common parent of all files and directories
    passed in `srcs`.

    If no directory in the tree contains a marker that would specify it's the
    project root, the root of the file system is returned.

    Returns a two-tuple with the first element as the project root path and
    the second element as a string describing the method by which the
    project root was discovered.
    """
    if stdin_filename is not None:
        srcs = tuple((stdin_filename if s == '-' else s for s in srcs))
    if not srcs:
        srcs = [str(_cached_resolve(Path.cwd()))]
    path_srcs = [_cached_resolve(Path(Path.cwd(), src)) for src in srcs]
    src_parents = [list(path.parents) + ([path] if path.is_dir() else []) for path in path_srcs]
    common_base = max(set.intersection(*(set(parents) for parents in src_parents)), key=lambda path: path.parts)
    for directory in (common_base, *common_base.parents):
        if (directory / '.git').exists():
            return (directory, '.git directory')
        if (directory / '.hg').is_dir():
            return (directory, '.hg directory')
        if (directory / 'pyproject.toml').is_file():
            pyproject_toml = _load_toml(directory / 'pyproject.toml')
            if 'black' in pyproject_toml.get('tool', {}):
                return (directory, 'pyproject.toml')
    return (directory, 'file system root')