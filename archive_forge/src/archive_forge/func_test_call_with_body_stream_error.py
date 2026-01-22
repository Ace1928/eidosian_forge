import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test_call_with_body_stream_error(self):
    """call_with_body_stream will abort the streamed body with an
        error if the stream raises an error during iteration.

        The resulting request will still be a complete message.
        """
    requester, output = self.make_client_encoder_and_output()
    requester.set_headers({})

    def stream_that_fails():
        yield b'aaa'
        yield b'bbb'
        raise Exception('Boom!')
    self.assertRaises(Exception, requester.call_with_body_stream, (b'one arg',), stream_that_fails())
    self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\x0bl7:one argeb\x00\x00\x00\x03aaab\x00\x00\x00\x03bbboEs\x00\x00\x00\tl5:erroree', output.getvalue())