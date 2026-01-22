import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def mkhead(repo: 'Repo', path: PathLike) -> 'Head':
    """:return: New branch/head instance"""
    return git.Head(repo, git.Head.to_full_path(path))