from __future__ import annotations
from glob import glob
import argparse
import errno
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import typing as T
import re
from . import build, environment
from .backend.backends import InstallData
from .mesonlib import (MesonException, Popen_safe, RealPathAction, is_windows,
from .scripts import depfixer, destdir_join
from .scripts.meson_exe import run_exe
def do_copydir(self, data: InstallData, src_dir: str, dst_dir: str, exclude: T.Optional[T.Tuple[T.Set[str], T.Set[str]]], install_mode: 'FileMode', dm: DirMaker, follow_symlinks: T.Optional[bool]=None) -> None:
    """
        Copies the contents of directory @src_dir into @dst_dir.

        For directory
            /foo/
              bar/
                excluded
                foobar
              file
        do_copydir(..., '/foo', '/dst/dir', {'bar/excluded'}) creates
            /dst/
              dir/
                bar/
                  foobar
                file

        Args:
            src_dir: str, absolute path to the source directory
            dst_dir: str, absolute path to the destination directory
            exclude: (set(str), set(str)), tuple of (exclude_files, exclude_dirs),
                     each element of the set is a path relative to src_dir.
        """
    if not os.path.isabs(src_dir):
        raise ValueError(f'src_dir must be absolute, got {src_dir}')
    if not os.path.isabs(dst_dir):
        raise ValueError(f'dst_dir must be absolute, got {dst_dir}')
    if exclude is not None:
        exclude_files, exclude_dirs = exclude
        exclude_files = {os.path.normpath(x) for x in exclude_files}
        exclude_dirs = {os.path.normpath(x) for x in exclude_dirs}
    else:
        exclude_files = exclude_dirs = set()
    for root, dirs, files in os.walk(src_dir):
        assert os.path.isabs(root)
        for d in dirs[:]:
            abs_src = os.path.join(root, d)
            filepart = os.path.relpath(abs_src, start=src_dir)
            abs_dst = os.path.join(dst_dir, filepart)
            if filepart in exclude_dirs:
                dirs.remove(d)
                continue
            if os.path.isdir(abs_dst):
                continue
            if os.path.exists(abs_dst):
                print(f'Tried to copy directory {abs_dst} but a file of that name already exists.')
                sys.exit(1)
            dm.makedirs(abs_dst)
            self.copystat(abs_src, abs_dst)
            self.sanitize_permissions(abs_dst, data.install_umask)
        for f in files:
            abs_src = os.path.join(root, f)
            filepart = os.path.relpath(abs_src, start=src_dir)
            if filepart in exclude_files:
                continue
            abs_dst = os.path.join(dst_dir, filepart)
            if os.path.isdir(abs_dst):
                print(f'Tried to copy file {abs_dst} but a directory of that name already exists.')
                sys.exit(1)
            parent_dir = os.path.dirname(abs_dst)
            if not os.path.isdir(parent_dir):
                dm.makedirs(parent_dir)
                self.copystat(os.path.dirname(abs_src), parent_dir)
            self.do_copyfile(abs_src, abs_dst, follow_symlinks=follow_symlinks)
            self.set_mode(abs_dst, install_mode, data.install_umask)