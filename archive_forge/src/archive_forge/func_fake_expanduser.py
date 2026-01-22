import signal
import sys
import threading
from _thread import interrupt_main  # type: ignore
from ... import builtins, config, errors, osutils
from ... import revision as _mod_revision
from ... import trace, transport, urlutils
from ...branch import Branch
from ...bzr.smart import client, medium
from ...bzr.smart.server import BzrServerFactory, SmartTCPServer
from ...controldir import ControlDir
from ...transport import remote
from .. import TestCaseWithMemoryTransport, TestCaseWithTransport
@staticmethod
def fake_expanduser(path):
    """A simple, environment-independent, function for the duration of this
        test.

        Paths starting with a path segment of '~user' will expand to start with
        '/home/user/'.  Every other path will be unchanged.
        """
    if path.split('/', 1)[0] == '~user':
        return '/home/user' + path[len('~user'):]
    return path