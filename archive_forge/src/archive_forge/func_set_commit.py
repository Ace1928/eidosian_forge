import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def set_commit(self, commit: Union[Commit, 'SymbolicReference', str], logmsg: Union[str, None]=None) -> 'SymbolicReference':
    """As set_object, but restricts the type of object to be a Commit.

        :raise ValueError: If commit is not a :class:`~git.objects.commit.Commit` object
            or doesn't point to a commit
        :return: self
        """
    invalid_type = False
    if isinstance(commit, Object):
        invalid_type = commit.type != Commit.type
    elif isinstance(commit, SymbolicReference):
        invalid_type = commit.object.type != Commit.type
    else:
        try:
            invalid_type = self.repo.rev_parse(commit).type != Commit.type
        except (BadObject, BadName) as e:
            raise ValueError('Invalid object: %s' % commit) from e
    if invalid_type:
        raise ValueError('Need commit, got %r' % commit)
    self.set_object(commit, logmsg)
    return self