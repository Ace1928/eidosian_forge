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
def install_symlinks(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for s in d.symlinks:
        if not self.should_install(s):
            continue
        full_dst_dir = get_destdir_path(destdir, fullprefix, s.install_path)
        full_link_name = get_destdir_path(destdir, fullprefix, s.name)
        dm.makedirs(full_dst_dir, exist_ok=True)
        if self.do_symlink(s.target, full_link_name, destdir, full_dst_dir, s.allow_missing):
            self.did_install_something = True