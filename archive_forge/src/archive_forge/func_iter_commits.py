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
def iter_commits(self, rev: Union[str, Commit, 'SymbolicReference', None]=None, paths: Union[PathLike, Sequence[PathLike]]='', **kwargs: Any) -> Iterator[Commit]:
    """A list of Commit objects representing the history of a given ref/commit.

        :param rev:
            Revision specifier, see git-rev-parse for viable options.
            If None, the active branch will be used.

        :param paths:
            An optional path or a list of paths; if set only commits that include the
            path or paths will be returned

        :param kwargs:
            Arguments to be passed to git-rev-list - common ones are max_count and skip.

        :note: To receive only commits between two named revisions, use the
            ``"revA...revB"`` revision specifier.

        :return: ``git.Commit[]``
        """
    if rev is None:
        rev = self.head.commit
    return Commit.iter_items(self, rev, paths, **kwargs)