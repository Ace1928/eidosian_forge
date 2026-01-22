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
def run_bzr_serve_then_func(self, serve_args, retcode=0, func=None, *func_args, **func_kwargs):
    """Run 'brz serve', and run the given func in a thread once the server
        has started.

        When 'func' terminates, the server will be terminated too.

        Returns stdout and stderr.
        """

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

    def on_server_start(backing_urls, tcp_server):
        t = threading.Thread(target=on_server_start_thread, args=(tcp_server,))
        t.start()
    SmartTCPServer.hooks.install_named_hook('server_started_ex', on_server_start, 'run_bzr_serve_then_func hook')
    self.overrideAttr(SmartTCPServer, '_ACCEPT_TIMEOUT', 0.1)
    try:
        out, err = self.run_bzr(['serve'] + list(serve_args), retcode=retcode)
    except KeyboardInterrupt as e:
        return (self._last_cmd_stdout.getvalue(), self._last_cmd_stderr.getvalue())
    return (out, err)