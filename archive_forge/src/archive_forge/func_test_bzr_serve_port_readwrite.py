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
def test_bzr_serve_port_readwrite(self):
    self.make_branch('.')
    process, url = self.start_server_port(['--allow-writes'])
    branch = Branch.open(url)
    self.make_read_requests(branch)
    self.assertServerFinishesCleanly(process)