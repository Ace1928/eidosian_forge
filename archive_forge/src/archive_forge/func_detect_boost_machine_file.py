from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
def detect_boost_machine_file(self, props: 'Properties') -> None:
    """Detect boost with values in the machine file or environment.

        The machine file values are defaulted to the environment values.
        """
    incdir = props.get('boost_includedir')
    assert incdir is None or isinstance(incdir, str)
    libdir = props.get('boost_librarydir')
    assert libdir is None or isinstance(libdir, str)
    if incdir and libdir:
        inc_dir = Path(incdir)
        lib_dir = Path(libdir)
        if not inc_dir.is_absolute() or not lib_dir.is_absolute():
            raise DependencyException('Paths given for boost_includedir and boost_librarydir in machine file must be absolute')
        mlog.debug('Trying to find boost with:')
        mlog.debug(f'  - boost_includedir = {inc_dir}')
        mlog.debug(f'  - boost_librarydir = {lib_dir}')
        return self.detect_split_root(inc_dir, lib_dir)
    elif incdir or libdir:
        raise DependencyException('Both boost_includedir *and* boost_librarydir have to be set in your machine file (one is not enough)')
    rootdir = props.get('boost_root')
    assert rootdir
    raw_paths = mesonlib.stringlistify(rootdir)
    paths = [Path(x) for x in raw_paths]
    if paths and any((not x.is_absolute() for x in paths)):
        raise DependencyException('boost_root path given in machine file must be absolute')
    self.check_and_set_roots(paths, use_system=False)