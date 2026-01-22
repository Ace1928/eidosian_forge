import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def find_parent_package(packages: List[str], package_dir: Mapping[str, str], root_dir: _Path) -> Optional[str]:
    """Find the parent package that is not a namespace."""
    packages = sorted(packages, key=len)
    common_ancestors = []
    for i, name in enumerate(packages):
        if not all((n.startswith(f'{name}.') for n in packages[i + 1:])):
            break
        common_ancestors.append(name)
    for name in common_ancestors:
        pkg_path = find_package_path(name, package_dir, root_dir)
        init = os.path.join(pkg_path, '__init__.py')
        if os.path.isfile(init):
            return name
    return None