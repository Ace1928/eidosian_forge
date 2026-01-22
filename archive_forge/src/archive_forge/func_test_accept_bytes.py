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
def test_accept_bytes(self):
    decoder = protocol.LengthPrefixedBodyDecoder()
    decoder.accept_bytes(b'')
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(6, decoder.next_read_size())
    self.assertEqual(b'', decoder.read_pending_data())
    self.assertEqual(b'', decoder.unused_data)
    decoder.accept_bytes(b'7')
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(6, decoder.next_read_size())
    self.assertEqual(b'', decoder.read_pending_data())
    self.assertEqual(b'', decoder.unused_data)
    decoder.accept_bytes(b'\na')
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(11, decoder.next_read_size())
    self.assertEqual(b'a', decoder.read_pending_data())
    self.assertEqual(b'', decoder.unused_data)
    decoder.accept_bytes(b'bcdefgd')
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(4, decoder.next_read_size())
    self.assertEqual(b'bcdefg', decoder.read_pending_data())
    self.assertEqual(b'', decoder.unused_data)
    decoder.accept_bytes(b'one')
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(1, decoder.next_read_size())
    self.assertEqual(b'', decoder.read_pending_data())
    self.assertEqual(b'', decoder.unused_data)
    decoder.accept_bytes(b'\nblarg')
    self.assertTrue(decoder.finished_reading)
    self.assertEqual(1, decoder.next_read_size())
    self.assertEqual(b'', decoder.read_pending_data())
    self.assertEqual(b'blarg', decoder.unused_data)