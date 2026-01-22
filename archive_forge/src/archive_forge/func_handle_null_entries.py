import contextlib
import datetime
import glob
from io import BytesIO
import os
from stat import S_ISLNK
import subprocess
import tempfile
from git.compat import (
from git.exc import GitCommandError, CheckoutError, GitError, InvalidGitRepositoryError
from git.objects import (
from git.objects.util import Serializable
from git.util import (
from gitdb.base import IStream
from gitdb.db import MemoryDB
import git.diff as git_diff
import os.path as osp
from .fun import (
from .typ import (
from .util import TemporaryFileSwap, post_clear_cache, default_index, git_working_dir
from typing import (
from git.types import Commit_ish, PathLike
@git_working_dir
def handle_null_entries(self: 'IndexFile') -> None:
    for ei in null_entries_indices:
        null_entry = entries[ei]
        new_entry = self._store_path(null_entry.path, fprogress)
        entries[ei] = BaseIndexEntry((null_entry.mode, new_entry.binsha, null_entry.stage, null_entry.path))