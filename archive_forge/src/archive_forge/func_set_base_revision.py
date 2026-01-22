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
def set_base_revision(self, revision_id, branch):
    """Set 'base' based on a branch and revision id

        :param revision_id: The revision to use for a tree
        :param branch: The branch containing this tree
        """
    self.base_rev_id = revision_id
    self.base_branch = branch
    self._maybe_fetch(branch, self.this_branch, revision_id)
    self.base_tree = self.revision_tree(revision_id)