from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def py_where(program: str, path: Optional[PathLike]=None) -> List[str]:
    """Perform a path search to assist :func:`is_cygwin_git`.

    This is not robust for general use. It is an implementation detail of
    :func:`is_cygwin_git`. When a search following all shell rules is needed,
    :func:`shutil.which` can be used instead.

    :note: Neither this function nor :func:`shutil.which` will predict the effect of an
        executable search on a native Windows system due to a :class:`subprocess.Popen`
        call without ``shell=True``, because shell and non-shell executable search on
        Windows differ considerably.
    """
    winprog_exts = _get_exe_extensions()

    def is_exec(fpath: str) -> bool:
        return osp.isfile(fpath) and os.access(fpath, os.X_OK) and (os.name != 'nt' or not winprog_exts or any((fpath.upper().endswith(ext) for ext in winprog_exts)))
    progs = []
    if not path:
        path = os.environ['PATH']
    for folder in str(path).split(os.pathsep):
        folder = folder.strip('"')
        if folder:
            exe_path = osp.join(folder, program)
            for f in [exe_path] + ['%s%s' % (exe_path, e) for e in winprog_exts]:
                if is_exec(f):
                    progs.append(f)
    return progs