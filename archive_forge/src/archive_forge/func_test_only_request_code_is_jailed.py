import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_only_request_code_is_jailed(self):
    transport = 'dummy transport'
    handler = request.SmartServerRequestHandler(transport, {b'foo': CheckJailRequest}, '/')
    handler.args_received((b'foo',))
    self.assertEqual(None, request.jail_info.transports)
    handler.accept_body(b'bytes')
    self.assertEqual(None, request.jail_info.transports)
    handler.end_received()
    self.assertEqual(None, request.jail_info.transports)
    self.assertEqual([[transport]] * 3, handler._command.jail_transports_log)