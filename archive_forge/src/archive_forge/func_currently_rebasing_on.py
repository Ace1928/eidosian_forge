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
def currently_rebasing_on(self) -> Commit | None:
    """
        :return: The commit which is currently being replayed while rebasing.

            None if we are not currently rebasing.
        """
    if self.git_dir:
        rebase_head_file = osp.join(self.git_dir, 'REBASE_HEAD')
    if not osp.isfile(rebase_head_file):
        return None
    with open(rebase_head_file, 'rt') as f:
        content = f.readline().strip()
    return self.commit(content)