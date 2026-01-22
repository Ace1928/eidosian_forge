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
def on_server_start_thread(tcp_server):
    """This runs concurrently with the server thread.

            The server is interrupted as soon as ``func`` finishes, even if an
            exception is encountered.
            """
    try:
        self.tcp_server = tcp_server
        if func is not None:
            try:
                func(*func_args, **func_kwargs)
            except Exception as e:
                trace.mutter('func broke: %r', e)
    finally:
        trace.mutter('interrupting...')
        interrupt_main()