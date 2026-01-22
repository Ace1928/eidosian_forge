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
def test__send_retries_on_write(self):
    output, vendor, smart_client = self.make_client_with_failing_medium()
    smart_request = client._SmartClientRequest(smart_client, b'hello', ())
    handler = smart_request._send(3)
    self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\tl5:helloee', output.getvalue())
    self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',), ('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes'])], vendor.calls)