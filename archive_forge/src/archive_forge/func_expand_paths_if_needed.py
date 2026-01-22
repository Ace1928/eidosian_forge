from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
def expand_paths_if_needed(paths, mode, num, fs, name_function):
    """Expand paths if they have a ``*`` in them (write mode) or any of ``*?[]``
    in them (read mode).

    :param paths: list of paths
    mode: str
        Mode in which to open files.
    num: int
        If opening in writing mode, number of files we expect to create.
    fs: filesystem object
    name_function: callable
        If opening in writing mode, this callable is used to generate path
        names. Names are generated for each partition by
        ``urlpath.replace('*', name_function(partition_index))``.
    :return: list of paths
    """
    expanded_paths = []
    paths = list(paths)
    if 'w' in mode:
        if sum([1 for p in paths if '*' in p]) > 1:
            raise ValueError('When writing data, only one filename mask can be specified.')
        num = max(num, len(paths))
        for curr_path in paths:
            if '*' in curr_path:
                expanded_paths.extend(_expand_paths(curr_path, name_function, num))
            else:
                expanded_paths.append(curr_path)
        if len(expanded_paths) > num:
            expanded_paths = expanded_paths[:num]
    else:
        for curr_path in paths:
            if has_magic(curr_path):
                expanded_paths.extend(fs.glob(curr_path))
            else:
                expanded_paths.append(curr_path)
    return expanded_paths