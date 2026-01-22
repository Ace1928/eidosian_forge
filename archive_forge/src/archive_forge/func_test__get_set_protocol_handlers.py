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
def test__get_set_protocol_handlers(self):
    handlers = transport._get_protocol_handlers()
    self.assertNotEqual([], handlers.keys())
    transport._clear_protocol_handlers()
    self.addCleanup(transport._set_protocol_handlers, handlers)
    self.assertEqual([], transport._get_protocol_handlers().keys())