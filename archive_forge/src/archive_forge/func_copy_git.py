from __future__ import annotations
import abc
import argparse
import gzip
import os
import sys
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import hashlib
import typing as T
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from mesonbuild.environment import Environment, detect_ninja
from mesonbuild.mesonlib import (MesonException, RealPathAction, get_meson_command, quiet_git,
from mesonbuild.msetup import add_arguments as msetup_argparse
from mesonbuild.wrap import wrap
from mesonbuild import mlog, build, coredata
from .scripts.meson_exe import run_exe
def copy_git(self, src: T.Union[str, os.PathLike], distdir: str, revision: str='HEAD', prefix: T.Optional[str]=None, subdir: T.Optional[str]=None) -> None:
    cmd = ['git', 'archive', '--format', 'tar', revision]
    if prefix is not None:
        cmd.insert(2, f'--prefix={prefix}/')
    if subdir is not None:
        cmd.extend(['--', subdir])
    with tempfile.TemporaryFile() as f:
        subprocess.check_call(cmd, cwd=src, stdout=f)
        f.seek(0)
        t = tarfile.open(fileobj=f)
        t.extractall(path=distdir)