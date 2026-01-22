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
def git_root(self, dir_: str) -> Path:
    prefix = quiet_git(['rev-parse', '--show-prefix'], dir_, check=True)[1].strip()
    if not prefix:
        return Path(dir_)
    prefix_level = len(Path(prefix).parents)
    return Path(dir_).parents[prefix_level - 1]