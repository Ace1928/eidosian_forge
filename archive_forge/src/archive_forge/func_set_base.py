import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def set_base(self, base_revision):
    """Set the base revision to use for the merge.

        :param base_revision: A 2-list containing a path and revision number.
        """
    trace.mutter('doing merge() with no base_revision specified')
    if base_revision == [None, None]:
        self.find_base()
    else:
        base_branch, self.base_tree = self._get_tree(base_revision)
        if base_revision[1] == -1:
            self.base_rev_id = base_branch.last_revision()
        elif base_revision[1] is None:
            self.base_rev_id = _mod_revision.NULL_REVISION
        else:
            self.base_rev_id = base_branch.get_rev_id(base_revision[1])
        self._maybe_fetch(base_branch, self.this_branch, self.base_rev_id)