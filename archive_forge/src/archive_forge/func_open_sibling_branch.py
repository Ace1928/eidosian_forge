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
def open_sibling_branch(control_dir, location, possible_transports=None):
    """Open a branch, possibly a sibling of another.

    :param control_dir: Control directory relative to which to lookup the
        location.
    :param location: Location to look up
    :return: branch to open
    """
    try:
        return control_dir.open_branch(location, possible_transports=possible_transports)
    except (errors.NotBranchError, controldir.NoColocatedBranchSupport):
        this_url = _get_branch_location(control_dir)
        return Branch.open(urlutils.join(this_url, '..', urlutils.escape(location)))