import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def log_append(self, oldbinsha: bytes, message: Union[str, None], newbinsha: Union[bytes, None]=None) -> 'RefLogEntry':
    """Append a logentry to the logfile of this ref.

        :param oldbinsha: Binary sha this ref used to point to.
        :param message: A message describing the change.
        :param newbinsha: The sha the ref points to now. If None, our current commit sha
            will be used.
        :return: The added :class:`~git.refs.log.RefLogEntry` instance.
        """
    try:
        committer_or_reader: Union['Actor', 'GitConfigParser'] = self.commit.committer
    except ValueError:
        committer_or_reader = self.repo.config_reader()
    if newbinsha is None:
        newbinsha = self.commit.binsha
    if message is None:
        message = ''
    return RefLog.append_entry(committer_or_reader, RefLog.path(self), oldbinsha, newbinsha, message)