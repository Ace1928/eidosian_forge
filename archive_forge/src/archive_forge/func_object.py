from .reference import Reference
from typing import Any, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike
@property
def object(self) -> Commit_ish:
    return Reference._get_object(self)