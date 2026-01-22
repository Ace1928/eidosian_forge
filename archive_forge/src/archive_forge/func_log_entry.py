import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def log_entry(self, index: int) -> 'RefLogEntry':
    """
        :return: RefLogEntry at the given index

        :param index: Python list compatible positive or negative index

        .. note:: This method must read part of the reflog during execution, hence
            it should be used sparingly, or only if you need just one index.
            In that case, it will be faster than the :meth:`log` method.
        """
    return RefLog.entry_at(RefLog.path(self), index)