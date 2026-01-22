from __future__ import annotations
import argparse, datetime, glob, json, os, platform, shutil, sys, tempfile, time
import cProfile as profile
from pathlib import Path
import typing as T
from . import build, coredata, environment, interpreter, mesonlib, mintro, mlog
from .mesonlib import MesonException
def validate_core_dirs(self, dir1: T.Optional[str], dir2: T.Optional[str]) -> T.Tuple[str, str]:
    invalid_msg_prefix = f'Neither source directory {dir1!r} nor build directory {dir2!r}'
    if dir1 is None:
        if dir2 is None:
            if not self.has_build_file('.') and self.has_build_file('..'):
                dir2 = '..'
            else:
                raise MesonException('Must specify at least one directory name.')
        dir1 = os.getcwd()
    if dir2 is None:
        dir2 = os.getcwd()
    ndir1 = os.path.abspath(os.path.realpath(dir1))
    ndir2 = os.path.abspath(os.path.realpath(dir2))
    if not os.path.exists(ndir1) and (not os.path.exists(ndir2)):
        raise MesonException(f'{invalid_msg_prefix} exist.')
    try:
        os.makedirs(ndir1, exist_ok=True)
    except FileExistsError as e:
        raise MesonException(f'{dir1} is not a directory') from e
    try:
        os.makedirs(ndir2, exist_ok=True)
    except FileExistsError as e:
        raise MesonException(f'{dir2} is not a directory') from e
    if os.path.samefile(ndir1, ndir2):
        has_undefined = any((s.st_ino == 0 and s.st_dev == 0 for s in (os.stat(ndir1), os.stat(ndir2))))
        if not has_undefined or ndir1 == ndir2:
            raise MesonException('Source and build directories must not be the same. Create a pristine build directory.')
    if self.has_build_file(ndir1):
        if self.has_build_file(ndir2):
            raise MesonException(f'Both directories contain a build file {environment.build_filename}.')
        return (ndir1, ndir2)
    if self.has_build_file(ndir2):
        return (ndir2, ndir1)
    raise MesonException(f'{invalid_msg_prefix} contain a build file {environment.build_filename}.')