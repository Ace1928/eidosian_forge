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
def gen_python_files(paths: Iterable[Path], root: Path, include: Pattern[str], exclude: Pattern[str], extend_exclude: Optional[Pattern[str]], force_exclude: Optional[Pattern[str]], report: Report, gitignore_dict: Optional[Dict[Path, PathSpec]], *, verbose: bool, quiet: bool) -> Iterator[Path]:
    """Generate all files under `path` whose paths are not excluded by the
    `exclude_regex`, `extend_exclude`, or `force_exclude` regexes,
    but are included by the `include` regex.

    Symbolic links pointing outside of the `root` directory are ignored.

    `report` is where output about exclusions goes.
    """
    assert root.is_absolute(), f'INTERNAL ERROR: `root` must be absolute but is {root}'
    for child in paths:
        assert child.is_absolute()
        root_relative_path = child.relative_to(root).as_posix()
        if gitignore_dict and _path_is_ignored(root_relative_path, root, gitignore_dict):
            report.path_ignored(child, 'matches a .gitignore file content')
            continue
        root_relative_path = '/' + root_relative_path
        if child.is_dir():
            root_relative_path += '/'
        if path_is_excluded(root_relative_path, exclude):
            report.path_ignored(child, 'matches the --exclude regular expression')
            continue
        if path_is_excluded(root_relative_path, extend_exclude):
            report.path_ignored(child, 'matches the --extend-exclude regular expression')
            continue
        if path_is_excluded(root_relative_path, force_exclude):
            report.path_ignored(child, 'matches the --force-exclude regular expression')
            continue
        if resolves_outside_root_or_cannot_stat(child, root, report):
            continue
        if child.is_dir():
            if gitignore_dict is not None:
                new_gitignore_dict = {**gitignore_dict, root / child: get_gitignore(child)}
            else:
                new_gitignore_dict = None
            yield from gen_python_files(child.iterdir(), root, include, exclude, extend_exclude, force_exclude, report, new_gitignore_dict, verbose=verbose, quiet=quiet)
        elif child.is_file():
            if child.suffix == '.ipynb' and (not jupyter_dependencies_are_installed(warn=verbose or not quiet)):
                continue
            include_match = include.search(root_relative_path) if include else True
            if include_match:
                yield child