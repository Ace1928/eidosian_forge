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
def test_get_base_path(self):
    """cmd_serve will turn the --directory option into a LocalTransport
        (optionally decorated with 'readonly+').  BzrServerFactory can
        determine the original --directory from that transport.
        """
    base_dir = osutils.abspath('/a/b/c') + '/'
    base_url = urlutils.local_path_to_url(base_dir) + '/'

    def capture_transport(transport, host, port, inet, timeout):
        self.bzr_serve_transport = transport
    cmd = builtins.cmd_serve()
    cmd.run(directory=base_dir, protocol=capture_transport)
    server_maker = BzrServerFactory()
    self.assertEqual('readonly+%s' % base_url, self.bzr_serve_transport.base)
    self.assertEqual(base_dir, server_maker.get_base_path(self.bzr_serve_transport))
    cmd.run(directory=base_dir, protocol=capture_transport, allow_writes=True)
    server_maker = BzrServerFactory()
    self.assertEqual(base_url, self.bzr_serve_transport.base)
    self.assertEqual(base_dir, server_maker.get_base_path(self.bzr_serve_transport))
    cmd.run(directory=base_url, protocol=capture_transport)
    server_maker = BzrServerFactory()
    self.assertEqual('readonly+%s' % base_url, self.bzr_serve_transport.base)
    self.assertEqual(base_dir, server_maker.get_base_path(self.bzr_serve_transport))