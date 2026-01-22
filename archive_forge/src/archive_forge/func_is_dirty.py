from __future__ import annotations
import gc
import logging
import os
import os.path as osp
from pathlib import Path
import re
import shlex
import warnings
import gitdb
from gitdb.db.loose import LooseObjectDB
from gitdb.exc import BadObject
from git.cmd import Git, handle_process_output
from git.compat import defenc, safe_decode
from git.config import GitConfigParser
from git.db import GitCmdObjectDB
from git.exc import (
from git.index import IndexFile
from git.objects import Submodule, RootModule, Commit
from git.refs import HEAD, Head, Reference, TagReference
from git.remote import Remote, add_progress, to_progress_instance
from git.util import (
from .fun import (
from git.types import (
from typing import (
from git.types import ConfigLevels_Tup, TypedDict
def is_dirty(self, index: bool=True, working_tree: bool=True, untracked_files: bool=False, submodules: bool=True, path: Optional[PathLike]=None) -> bool:
    """
        :return:
            ``True`` if the repository is considered dirty. By default it will react
            like a git-status without untracked files, hence it is dirty if the
            index or the working copy have changes.
        """
    if self._bare:
        return False
    default_args = ['--abbrev=40', '--full-index', '--raw']
    if not submodules:
        default_args.append('--ignore-submodules')
    if path:
        default_args.extend(['--', str(path)])
    if index:
        if osp.isfile(self.index.path) and len(self.git.diff('--cached', *default_args)):
            return True
    if working_tree:
        if len(self.git.diff(*default_args)):
            return True
    if untracked_files:
        if len(self._get_untracked_files(path, ignore_submodules=not submodules)):
            return True
    return False