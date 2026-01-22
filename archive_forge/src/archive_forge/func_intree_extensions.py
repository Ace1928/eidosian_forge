import contextlib
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
import distutils.ccompiler
import distutils.errors
def intree_extensions(paths: Iterable[str], package_dir: Optional[Dict[str, str]]=None) -> List[Pybind11Extension]:
    """
    Generate Pybind11Extensions from source files directly located in a Python
    source tree.

    ``package_dir`` behaves as in ``setuptools.setup``.  If unset, the Python
    package root parent is determined as the first parent directory that does
    not contain an ``__init__.py`` file.
    """
    exts = []
    if package_dir is None:
        for path in paths:
            parent, _ = os.path.split(path)
            while os.path.exists(os.path.join(parent, '__init__.py')):
                parent, _ = os.path.split(parent)
            relname, _ = os.path.splitext(os.path.relpath(path, parent))
            qualified_name = relname.replace(os.path.sep, '.')
            exts.append(Pybind11Extension(qualified_name, [path]))
        return exts
    for path in paths:
        for prefix, parent in package_dir.items():
            if path.startswith(parent):
                relname, _ = os.path.splitext(os.path.relpath(path, parent))
                qualified_name = relname.replace(os.path.sep, '.')
                if prefix:
                    qualified_name = prefix + '.' + qualified_name
                exts.append(Pybind11Extension(qualified_name, [path]))
                break
        else:
            msg = f"path {path} is not a child of any of the directories listed in 'package_dir' ({package_dir})"
            raise ValueError(msg)
    return exts