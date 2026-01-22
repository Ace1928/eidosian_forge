import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def find_package_path(name: str, package_dir: Mapping[str, str], root_dir: _Path) -> str:
    """Given a package name, return the path where it should be found on
    disk, considering the ``package_dir`` option.

    >>> path = find_package_path("my.pkg", {"": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested/my/pkg'

    >>> path = find_package_path("my.pkg", {"my": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested/pkg'

    >>> path = find_package_path("my.pkg", {"my.pkg": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested'

    >>> path = find_package_path("other.pkg", {"my.pkg": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './other/pkg'
    """
    parts = name.split('.')
    for i in range(len(parts), 0, -1):
        partial_name = '.'.join(parts[:i])
        if partial_name in package_dir:
            parent = package_dir[partial_name]
            return os.path.join(root_dir, parent, *parts[i:])
    parent = package_dir.get('') or ''
    return os.path.join(root_dir, *parent.split('/'), *parts)