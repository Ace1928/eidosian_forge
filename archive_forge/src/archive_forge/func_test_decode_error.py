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
def test_decode_error(self):
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    chunk_one = b'b\nfirst chunk'
    error_signal = b'ERR\n'
    error_chunks = b'5\npart1' + b'5\npart2'
    finish = b'END\n'
    decoder.accept_bytes(chunk_one + error_signal + error_chunks + finish)
    self.assertTrue(decoder.finished_reading)
    self.assertEqual(b'first chunk', decoder.read_next_chunk())
    expected_failure = _mod_request.FailedSmartServerResponse((b'part1', b'part2'))
    self.assertEqual(expected_failure, decoder.read_next_chunk())