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
def test_send_response_with_body_stream_buffers_writes(self):
    """A normal response with a stream body writes to the medium once."""
    response = _mod_request.SuccessfulSmartServerResponse((b'arg', b'arg'), body_stream=[b'chunk1', b'chunk2'])
    self.responder.send_response(response)
    self.assertWriteCount(3)