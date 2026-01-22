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
def test_decorator(self):
    t = transport.get_transport_from_url('trace+memory://')
    self.assertIsInstance(t, breezy.transport.trace.TransportTraceDecorator)