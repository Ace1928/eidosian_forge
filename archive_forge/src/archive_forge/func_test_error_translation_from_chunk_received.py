import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_error_translation_from_chunk_received(self):
    handler = request.SmartServerRequestHandler(None, {b'foo': ChunkErrorRequest}, '/')
    handler.args_received((b'foo',))
    self.assertNoResponse(handler)
    handler.accept_body(b'bytes')
    self.assertResponseIsTranslatedError(handler)