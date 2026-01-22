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
def test_clone_preserves_activity(self):
    t = transport.get_transport_from_url('trace+memory://')
    t2 = t.clone('.')
    self.assertTrue(t is not t2)
    self.assertTrue(t._activity is t2._activity)