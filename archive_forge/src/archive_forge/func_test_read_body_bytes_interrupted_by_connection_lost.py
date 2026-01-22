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
def test_read_body_bytes_interrupted_by_connection_lost(self):
    interrupted_body_stream = b'oSs\x00\x00\x00\x02leb\x00\x00\xff\xffincomplete chunk'
    response_handler = self.make_response_handler(interrupted_body_stream)
    self.assertRaises(errors.ConnectionReset, response_handler.read_body_bytes)