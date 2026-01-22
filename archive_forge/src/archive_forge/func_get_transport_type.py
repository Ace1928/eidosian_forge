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
def get_transport_type(typestring):
    """Parse and return a transport specifier."""
    if typestring == 'sftp':
        from .tests import stub_sftp
        return stub_sftp.SFTPAbsoluteServer
    elif typestring == 'memory':
        from .tests import test_server
        return memory.MemoryServer
    elif typestring == 'fakenfs':
        from .tests import test_server
        return test_server.FakeNFSServer
    msg = 'No known transport type %s. Supported types are: sftp\n' % typestring
    raise errors.CommandError(msg)