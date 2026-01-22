from __future__ import annotations
import os
import stat
from pathlib import Path
from string import digits
from git.exc import WorkTreeRepositoryUnsupported
from git.objects import Object
from git.refs import SymbolicReference
from git.util import hex_to_bin, bin_to_hex, cygpath
from gitdb.exc import (
import os.path as osp
from git.cmd import Git
from typing import Union, Optional, cast, TYPE_CHECKING
from git.types import Commit_ish
def find_worktree_git_dir(dotgit: 'PathLike') -> Optional[str]:
    """Search for a gitdir for this worktree."""
    try:
        statbuf = os.stat(dotgit)
    except OSError:
        return None
    if not stat.S_ISREG(statbuf.st_mode):
        return None
    try:
        lines = Path(dotgit).read_text().splitlines()
        for key, value in [line.strip().split(': ') for line in lines]:
            if key == 'gitdir':
                return value
    except ValueError:
        pass
    return None