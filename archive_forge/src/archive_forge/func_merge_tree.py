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
@post_clear_cache
@default_index
def merge_tree(self, rhs: Treeish, base: Union[None, Treeish]=None) -> 'IndexFile':
    """Merge the given rhs treeish into the current index, possibly taking
        a common base treeish into account.

        As opposed to the :func:`IndexFile.from_tree` method, this allows you to use an
        already existing tree as the left side of the merge.

        :param rhs:
            Treeish reference pointing to the 'other' side of the merge.

        :param base:
            Optional treeish reference pointing to the common base of 'rhs' and this
            index which equals lhs.

        :return:
            self (containing the merge and possibly unmerged entries in case of
            conflicts)

        :raise GitCommandError:
            If there is a merge conflict. The error will be raised at the first
            conflicting path. If you want to have proper merge resolution to be done by
            yourself, you have to commit the changed index (or make a valid tree from
            it) and retry with a three-way index.from_tree call.
        """
    args: List[Union[Treeish, str]] = ['--aggressive', '-i', '-m']
    if base is not None:
        args.append(base)
    args.append(rhs)
    self.repo.git.read_tree(args)
    return self