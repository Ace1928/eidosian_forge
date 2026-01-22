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
def make_test_server(self, base_path='/'):
    """Make and start a BzrServerFactory, backed by a memory transport, and
        creat '/home/user' in that transport.
        """
    bzr_server = BzrServerFactory(self.fake_expanduser, lambda t: base_path)
    mem_transport = self.get_transport()
    mem_transport.mkdir('home')
    mem_transport.mkdir('home/user')
    bzr_server.set_up(mem_transport, None, None, inet=True, timeout=4.0)
    self.addCleanup(bzr_server.tear_down)
    return bzr_server