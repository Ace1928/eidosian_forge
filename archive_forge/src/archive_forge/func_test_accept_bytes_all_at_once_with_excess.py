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
def test_accept_bytes_all_at_once_with_excess(self):
    decoder = protocol.LengthPrefixedBodyDecoder()
    decoder.accept_bytes(b'1\nadone\nunused')
    self.assertTrue(decoder.finished_reading)
    self.assertEqual(1, decoder.next_read_size())
    self.assertEqual(b'a', decoder.read_pending_data())
    self.assertEqual(b'unused', decoder.unused_data)