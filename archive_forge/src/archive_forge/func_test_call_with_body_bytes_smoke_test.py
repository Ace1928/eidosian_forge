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
def test_call_with_body_bytes_smoke_test(self):
    """A smoke test for ProtocolThreeRequester.call_with_body_bytes.

        This test checks that a particular simple invocation of
        call_with_body_bytes emits the correct bytes for that invocation.
        """
    requester, output = self.make_client_encoder_and_output()
    requester.set_headers({b'header name': b'header value'})
    requester.call_with_body_bytes((b'one arg',), b'body bytes')
    self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x1fd11:header name12:header valuees\x00\x00\x00\x0bl7:one argeb\x00\x00\x00\nbody bytese', output.getvalue())