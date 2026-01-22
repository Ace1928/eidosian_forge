from git.config import GitConfigParser, SectionConstraint
from git.util import join_path
from git.exc import GitCommandError
from .symbolic import SymbolicReference
from .reference import Reference
from typing import Any, Sequence, Union, TYPE_CHECKING
from git.types import PathLike, Commit_ish
def strip_quotes(string: str) -> str:
    if string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    return string