import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
def find_packages(*, namespaces=True, fill_package_dir: Optional[Dict[str, str]]=None, root_dir: Optional[_Path]=None, **kwargs) -> List[str]:
    """Works similarly to :func:`setuptools.find_packages`, but with all
    arguments given as keyword arguments. Moreover, ``where`` can be given
    as a list (the results will be simply concatenated).

    When the additional keyword argument ``namespaces`` is ``True``, it will
    behave like :func:`setuptools.find_namespace_packages`` (i.e. include
    implicit namespaces as per :pep:`420`).

    The ``where`` argument will be considered relative to ``root_dir`` (or the current
    working directory when ``root_dir`` is not given).

    If the ``fill_package_dir`` argument is passed, this function will consider it as a
    similar data structure to the ``package_dir`` configuration parameter add fill-in
    any missing package location.

    :rtype: list
    """
    from setuptools.discovery import construct_package_dir
    from setuptools.extern.more_itertools import unique_everseen, always_iterable
    if namespaces:
        from setuptools.discovery import PEP420PackageFinder as PackageFinder
    else:
        from setuptools.discovery import PackageFinder
    root_dir = root_dir or os.curdir
    where = kwargs.pop('where', ['.'])
    packages: List[str] = []
    fill_package_dir = {} if fill_package_dir is None else fill_package_dir
    search = list(unique_everseen(always_iterable(where)))
    if len(search) == 1 and all((not _same_path(search[0], x) for x in ('.', root_dir))):
        fill_package_dir.setdefault('', search[0])
    for path in search:
        package_path = _nest_path(root_dir, path)
        pkgs = PackageFinder.find(package_path, **kwargs)
        packages.extend(pkgs)
        if pkgs and (not (fill_package_dir.get('') == path or os.path.samefile(package_path, root_dir))):
            fill_package_dir.update(construct_package_dir(pkgs, path))
    return packages