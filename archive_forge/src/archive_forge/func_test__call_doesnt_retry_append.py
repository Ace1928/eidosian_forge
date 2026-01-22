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
def test__call_doesnt_retry_append(self):
    response = self.make_response(('appended', b'8'))
    output, vendor, smart_client = self.make_client_with_failing_medium(fail_at_write=False, response=response)
    smart_request = client._SmartClientRequest(smart_client, b'append', (b'foo', b''), body=b'content\n')
    self.assertRaises(errors.ConnectionReset, smart_request._call, 3)