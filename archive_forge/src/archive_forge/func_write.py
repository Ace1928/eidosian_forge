import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def write(self) -> None:
    rval: None = super().write()
    self.flush_to_index()
    return rval