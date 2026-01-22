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
def iter_blobs(self, predicate: Callable[[Tuple[StageType, Blob]], bool]=lambda t: True) -> Iterator[Tuple[StageType, Blob]]:
    """
        :return: Iterator yielding tuples of Blob objects and stages, tuple(stage, Blob)

        :param predicate:
            Function(t) returning True if tuple(stage, Blob) should be yielded by the
            iterator. A default filter, the BlobFilter, allows you to yield blobs
            only if they match a given list of paths.
        """
    for entry in self.entries.values():
        blob = entry.to_blob(self.repo)
        blob.size = entry.size
        output = (entry.stage, blob)
        if predicate(output):
            yield output