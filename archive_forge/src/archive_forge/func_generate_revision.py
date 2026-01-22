from __future__ import annotations
from contextlib import contextmanager
import datetime
import os
import re
import shutil
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import revision
from . import write_hooks
from .. import util
from ..runtime import migration
from ..util import compat
from ..util import not_none
def generate_revision(self, revid: str, message: Optional[str], head: Optional[_RevIdType]=None, splice: Optional[bool]=False, branch_labels: Optional[_RevIdType]=None, version_path: Optional[str]=None, depends_on: Optional[_RevIdType]=None, **kw: Any) -> Optional[Script]:
    """Generate a new revision file.

        This runs the ``script.py.mako`` template, given
        template arguments, and creates a new file.

        :param revid: String revision id.  Typically this
         comes from ``alembic.util.rev_id()``.
        :param message: the revision message, the one passed
         by the -m argument to the ``revision`` command.
        :param head: the head revision to generate against.  Defaults
         to the current "head" if no branches are present, else raises
         an exception.
        :param splice: if True, allow the "head" version to not be an
         actual head; otherwise, the selected head must be a head
         (e.g. endpoint) revision.

        """
    if head is None:
        head = 'head'
    try:
        Script.verify_rev_id(revid)
    except revision.RevisionError as err:
        raise util.CommandError(err.args[0]) from err
    with self._catch_revision_errors(multiple_heads='Multiple heads are present; please specify the head revision on which the new revision should be based, or perform a merge.'):
        heads = cast(Tuple[Optional['Revision'], ...], self.revision_map.get_revisions(head))
        for h in heads:
            assert h != 'base'
    if len(set(heads)) != len(heads):
        raise util.CommandError('Duplicate head revisions specified')
    create_date = self._generate_create_date()
    if version_path is None:
        if len(self._version_locations) > 1:
            for head_ in heads:
                if head_ is not None:
                    assert isinstance(head_, Script)
                    version_path = os.path.dirname(head_.path)
                    break
            else:
                raise util.CommandError('Multiple version locations present, please specify --version-path')
        else:
            version_path = self.versions
    norm_path = os.path.normpath(os.path.abspath(version_path))
    for vers_path in self._version_locations:
        if os.path.normpath(vers_path) == norm_path:
            break
    else:
        raise util.CommandError('Path %s is not represented in current version locations' % version_path)
    if self.version_locations:
        self._ensure_directory(version_path)
    path = self._rev_path(version_path, revid, message, create_date)
    if not splice:
        for head_ in heads:
            if head_ is not None and (not head_.is_head):
                raise util.CommandError('Revision %s is not a head revision; please specify --splice to create a new branch from this revision' % head_.revision)
    resolved_depends_on: Optional[List[str]]
    if depends_on:
        with self._catch_revision_errors():
            resolved_depends_on = [dep if dep in rev.branch_labels else rev.revision for rev, dep in [(not_none(self.revision_map.get_revision(dep)), dep) for dep in util.to_list(depends_on)]]
    else:
        resolved_depends_on = None
    self._generate_template(os.path.join(self.dir, 'script.py.mako'), path, up_revision=str(revid), down_revision=revision.tuple_rev_as_scalar(tuple((h.revision if h is not None else None for h in heads))), branch_labels=util.to_tuple(branch_labels), depends_on=revision.tuple_rev_as_scalar(resolved_depends_on), create_date=create_date, comma=util.format_as_comma, message=message if message is not None else 'empty message', **kw)
    post_write_hooks = self.hook_config
    if post_write_hooks:
        write_hooks._run_hooks(path, post_write_hooks)
    try:
        script = Script._from_path(self, path)
    except revision.RevisionError as err:
        raise util.CommandError(err.args[0]) from err
    if script is None:
        return None
    if branch_labels and (not script.branch_labels):
        raise util.CommandError("Version %s specified branch_labels %s, however the migration file %s does not have them; have you upgraded your script.py.mako to include the 'branch_labels' section?" % (script.revision, branch_labels, script.path))
    self.revision_map.add_revision(script)
    return script