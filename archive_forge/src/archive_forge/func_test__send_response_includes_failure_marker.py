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
def test__send_response_includes_failure_marker(self):
    """FailedSmartServerResponse have 'failed
' after the version."""
    out_stream = BytesIO()
    smart_protocol = protocol.SmartServerRequestProtocolTwo(None, out_stream.write)
    smart_protocol._send_response(_mod_request.FailedSmartServerResponse((b'x',)))
    self.assertEqual(protocol.RESPONSE_VERSION_TWO + b'failed\nx\n', out_stream.getvalue())