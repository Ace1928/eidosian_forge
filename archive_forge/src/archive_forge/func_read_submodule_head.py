import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def read_submodule_head(path: Union[str, bytes]) -> Optional[bytes]:
    """Read the head commit of a submodule.

    Args:
      path: path to the submodule
    Returns: HEAD sha, None if not a valid head/repository
    """
    from .errors import NotGitRepository
    from .repo import Repo
    if not isinstance(path, str):
        path = os.fsdecode(path)
    try:
        repo = Repo(path)
    except NotGitRepository:
        return None
    try:
        return repo.head()
    except KeyError:
        return None