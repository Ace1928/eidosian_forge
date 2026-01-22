import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@property
def is_detached(self) -> bool:
    """
        :return:
            True if we are a detached reference, hence we point to a specific commit
            instead to another reference.
        """
    try:
        self.ref
        return False
    except TypeError:
        return True