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
def find_submodule_git_dir(d: 'PathLike') -> Optional['PathLike']:
    """Search for a submodule repo."""
    if is_git_dir(d):
        return d
    try:
        with open(d) as fp:
            content = fp.read().rstrip()
    except IOError:
        pass
    else:
        if content.startswith('gitdir: '):
            path = content[8:]
            if Git.is_cygwin():
                path = cygpath(path)
            if not osp.isabs(path):
                path = osp.normpath(osp.join(osp.dirname(d), path))
            return find_submodule_git_dir(path)
    return None