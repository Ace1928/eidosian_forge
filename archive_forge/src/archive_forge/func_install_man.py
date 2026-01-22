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
def install_man(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for m in d.man:
        if not self.should_install(m):
            continue
        full_source_filename = m.path
        outfilename = get_destdir_path(destdir, fullprefix, m.install_path)
        outdir = os.path.dirname(outfilename)
        if self.do_copyfile(full_source_filename, outfilename, makedirs=(dm, outdir)):
            self.did_install_something = True
        self.set_mode(outfilename, m.install_mode, d.install_umask)