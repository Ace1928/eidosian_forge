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
def handle_stderr(proc: 'Popen[bytes]', iter_checked_out_files: Iterable[PathLike]) -> None:
    stderr_IO = proc.stderr
    if not stderr_IO:
        return
    stderr_bytes = stderr_IO.read()
    stderr = stderr_bytes.decode(defenc)
    endings = (' already exists', ' is not in the cache', ' does not exist at stage', ' is unmerged')
    for line in stderr.splitlines():
        if not line.startswith('git checkout-index: ') and (not line.startswith('git-checkout-index: ')):
            is_a_dir = ' is a directory'
            unlink_issue = "unable to unlink old '"
            already_exists_issue = ' already exists, no checkout'
            if line.endswith(is_a_dir):
                failed_files.append(line[:-len(is_a_dir)])
                failed_reasons.append(is_a_dir)
            elif line.startswith(unlink_issue):
                failed_files.append(line[len(unlink_issue):line.rfind("'")])
                failed_reasons.append(unlink_issue)
            elif line.endswith(already_exists_issue):
                failed_files.append(line[:-len(already_exists_issue)])
                failed_reasons.append(already_exists_issue)
            else:
                unknown_lines.append(line)
            continue
        for e in endings:
            if line.endswith(e):
                failed_files.append(line[20:-len(e)])
                failed_reasons.append(e)
                break
    if unknown_lines:
        raise GitCommandError(('git-checkout-index',), 128, stderr)
    if failed_files:
        valid_files = list(set(iter_checked_out_files) - set(failed_files))
        raise CheckoutError('Some files could not be checked out from the index due to local modifications', failed_files, valid_files, failed_reasons)