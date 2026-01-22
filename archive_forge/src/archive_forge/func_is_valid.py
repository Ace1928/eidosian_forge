import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def is_valid(self) -> bool:
    """
        :return:
            True if the reference is valid, hence it can be read and points to
            a valid object or reference.
        """
    try:
        self.object
    except (OSError, ValueError):
        return False
    else:
        return True