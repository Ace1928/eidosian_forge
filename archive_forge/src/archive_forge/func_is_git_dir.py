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
def is_git_dir(d: 'PathLike') -> bool:
    """This is taken from the git setup.c:is_git_directory
    function.

    @throws WorkTreeRepositoryUnsupported if it sees a worktree directory. It's quite hacky to do that here,
            but at least clearly indicates that we don't support it.
            There is the unlikely danger to throw if we see directories which just look like a worktree dir,
            but are none."""
    if osp.isdir(d):
        if (osp.isdir(osp.join(d, 'objects')) or 'GIT_OBJECT_DIRECTORY' in os.environ) and osp.isdir(osp.join(d, 'refs')):
            headref = osp.join(d, 'HEAD')
            return osp.isfile(headref) or (osp.islink(headref) and os.readlink(headref).startswith('refs'))
        elif osp.isfile(osp.join(d, 'gitdir')) and osp.isfile(osp.join(d, 'commondir')) and osp.isfile(osp.join(d, 'gitfile')):
            raise WorkTreeRepositoryUnsupported(d)
    return False