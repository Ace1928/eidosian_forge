import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def remove_nested_packages(packages: List[str]) -> List[str]:
    """Remove nested packages from a list of packages.

    >>> remove_nested_packages(["a", "a.b1", "a.b2", "a.b1.c1"])
    ['a']
    >>> remove_nested_packages(["a", "b", "c.d", "c.d.e.f", "g.h", "a.a1"])
    ['a', 'b', 'c.d', 'g.h']
    """
    pkgs = sorted(packages, key=len)
    top_level = pkgs[:]
    size = len(pkgs)
    for i, name in enumerate(reversed(pkgs)):
        if any((name.startswith(f'{other}.') for other in top_level)):
            top_level.pop(size - i - 1)
    return top_level