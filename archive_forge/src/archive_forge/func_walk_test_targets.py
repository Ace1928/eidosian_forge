from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def walk_test_targets(path: t.Optional[str]=None, module_path: t.Optional[str]=None, extensions: t.Optional[tuple[str, ...]]=None, prefix: t.Optional[str]=None, extra_dirs: t.Optional[tuple[str, ...]]=None, include_symlinks: bool=False, include_symlinked_directories: bool=False) -> c.Iterable[TestTarget]:
    """Iterate over available test targets."""
    if path:
        file_paths = data_context().content.walk_files(path, include_symlinked_directories=include_symlinked_directories)
    else:
        file_paths = data_context().content.all_files(include_symlinked_directories=include_symlinked_directories)
    for file_path in file_paths:
        name, ext = os.path.splitext(os.path.basename(file_path))
        if extensions and ext not in extensions:
            continue
        if prefix and (not name.startswith(prefix)):
            continue
        symlink = os.path.islink(to_bytes(file_path.rstrip(os.path.sep)))
        if symlink and (not include_symlinks):
            continue
        yield TestTarget(to_text(file_path), module_path, prefix, path, symlink)
    file_paths = []
    if extra_dirs:
        for extra_dir in extra_dirs:
            for file_path in data_context().content.get_files(extra_dir):
                file_paths.append(file_path)
    for file_path in file_paths:
        symlink = os.path.islink(to_bytes(file_path.rstrip(os.path.sep)))
        if symlink and (not include_symlinks):
            continue
        yield TestTarget(file_path, module_path, prefix, path, symlink)