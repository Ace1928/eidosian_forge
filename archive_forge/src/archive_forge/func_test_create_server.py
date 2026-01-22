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
def test_create_server(self):
    server = memory.MemoryServer()
    server.start_server()
    url = server.get_url()
    self.assertTrue(url in transport.transport_list_registry)
    t = transport.get_transport_from_url(url)
    del t
    server.stop_server()
    self.assertFalse(url in transport.transport_list_registry)
    self.assertRaises(UnsupportedProtocol, transport.get_transport, url)