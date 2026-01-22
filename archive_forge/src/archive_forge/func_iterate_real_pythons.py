from __future__ import annotations
import collections.abc as c
import json
import os
import pathlib
import sys
import typing as t
from .config import (
from .util import (
from .util_common import (
from .host_configs import (
from .python_requirements import (
def iterate_real_pythons(version: str) -> c.Iterable[str]:
    """
    Iterate through available real python interpreters of the requested version.
    The current interpreter will be checked and then the path will be searched.
    """
    version_info = str_to_version(version)
    current_python = None
    if version_info == sys.version_info[:len(version_info)]:
        current_python = sys.executable
        real_prefix = get_python_real_prefix(current_python)
        if real_prefix:
            current_python = find_python(version, os.path.join(real_prefix, 'bin'))
        if current_python:
            yield current_python
    path = os.environ.get('PATH', os.path.defpath)
    if not path:
        return
    found_python = find_python(version, path)
    if not found_python:
        return
    if found_python == current_python:
        return
    real_prefix = get_python_real_prefix(found_python)
    if real_prefix:
        found_python = find_python(version, os.path.join(real_prefix, 'bin'))
    if found_python:
        yield found_python