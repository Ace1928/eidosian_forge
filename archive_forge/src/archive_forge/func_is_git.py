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
def is_git(src_root: str) -> bool:
    """
    Checks if meson.build file at the root source directory is tracked by git.
    It could be a subproject part of the parent project git repository.
    """
    return quiet_git(['ls-files', '--error-unmatch', 'meson.build'], src_root)[0]