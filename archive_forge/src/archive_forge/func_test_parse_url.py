import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_parse_url(self):
    t = transport.ConnectedTransport('http://simple.example.com/home/source')
    self.assertEqual(t._parsed_url.host, 'simple.example.com')
    self.assertEqual(t._parsed_url.port, None)
    self.assertEqual(t._parsed_url.path, '/home/source/')
    self.assertTrue(t._parsed_url.user is None)
    self.assertTrue(t._parsed_url.password is None)
    self.assertEqual(t.base, 'http://simple.example.com/home/source/')