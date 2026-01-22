import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
def get_merger_from_uncommitted(self, tree, location, pb):
    """Get a merger for uncommitted changes.

        :param tree: The tree the merger should apply to.
        :param location: The location containing uncommitted changes.
        :param pb: The progress bar to use for showing progress.
        """
    location = self._select_branch_location(tree, location)[0]
    other_tree, other_path = WorkingTree.open_containing(location)
    merger = _mod_merge.Merger.from_uncommitted(tree, other_tree, pb)
    if other_path != '':
        merger.interesting_files = [other_path]
    return merger