import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def set_object(self, object: Union[Commit_ish, 'SymbolicReference', str], logmsg: Union[str, None]=None) -> 'SymbolicReference':
    """Set the object we point to, possibly dereference our symbolic reference first.
        If the reference does not exist, it will be created.

        :param object: A refspec, a :class:`SymbolicReference` or an
            :class:`~git.objects.base.Object` instance.
            :class:`SymbolicReference` instances will be dereferenced beforehand to
            obtain the object they point to.
        :param logmsg: If not None, the message will be used in the reflog entry to be
            written. Otherwise the reflog is not altered.
        :note: Plain :class:`SymbolicReference` instances may not actually point to
            objects by convention.
        :return: self
        """
    if isinstance(object, SymbolicReference):
        object = object.object
    is_detached = True
    try:
        is_detached = self.is_detached
    except ValueError:
        pass
    if is_detached:
        return self.set_reference(object, logmsg)
    return self._get_reference().set_object(object, logmsg)