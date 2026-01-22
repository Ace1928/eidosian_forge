import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
@property
def stale_refs(self) -> IterableList[Reference]:
    """
        :return:
            IterableList RemoteReference objects that do not have a corresponding
            head in the remote reference anymore as they have been deleted on the
            remote side, but are still available locally.

            The IterableList is prefixed, hence the 'origin' must be omitted. See
            'refs' property for an example.

            To make things more complicated, it can be possible for the list to include
            other kinds of references, for example, tag references, if these are stale
            as well. This is a fix for the issue described here:
            https://github.com/gitpython-developers/GitPython/issues/260
        """
    out_refs: IterableList[Reference] = IterableList(RemoteReference._id_attribute_, '%s/' % self.name)
    for line in self.repo.git.remote('prune', '--dry-run', self).splitlines()[2:]:
        token = ' * [would prune] '
        if not line.startswith(token):
            continue
        ref_name = line.replace(token, '')
        if ref_name.startswith(Reference._common_path_default + '/'):
            out_refs.append(Reference.from_path(self.repo, ref_name))
        else:
            fqhn = '%s/%s' % (RemoteReference._common_path_default, ref_name)
            out_refs.append(RemoteReference(self.repo, fqhn))
    return out_refs