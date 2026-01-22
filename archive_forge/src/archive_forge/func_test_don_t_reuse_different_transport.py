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
def test_don_t_reuse_different_transport(self):
    t1 = transport.get_transport_from_url('http://foo/path')
    t2 = transport.get_transport_from_url('http://bar/path', possible_transports=[t1])
    self.assertIsNot(t1, t2)