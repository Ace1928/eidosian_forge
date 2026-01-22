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
def test_streamed_body_bytes(self):
    body_header = b'chunked\n'
    two_body_chunks = b'4\n1234' + b'3\n567'
    body_terminator = b'END\n'
    server_bytes = protocol.RESPONSE_VERSION_TWO + b'success\nok\n' + body_header + two_body_chunks + body_terminator
    input = BytesIO(server_bytes)
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
    request = client_medium.get_request()
    smart_protocol = protocol.SmartClientRequestProtocolTwo(request)
    smart_protocol.call(b'foo')
    smart_protocol.read_response_tuple(True)
    stream = smart_protocol.read_streamed_body()
    self.assertEqual([b'1234', b'567'], list(stream))