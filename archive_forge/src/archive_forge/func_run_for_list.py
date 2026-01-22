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
def run_for_list(self, directory=None):
    if directory is None:
        directory = '.'
    tree = WorkingTree.open_containing(directory)[0]
    self.enter_context(tree.lock_read())
    manager = tree.get_shelf_manager()
    shelves = manager.active_shelves()
    if len(shelves) == 0:
        note(gettext('No shelved changes.'))
        return 0
    for shelf_id in reversed(shelves):
        message = manager.get_metadata(shelf_id).get(b'message')
        if message is None:
            message = '<no message>'
        self.outf.write('%3d: %s\n' % (shelf_id, message))
    return 1