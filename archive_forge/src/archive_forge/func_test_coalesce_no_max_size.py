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
def test_coalesce_no_max_size(self):
    self.check([(10, 170, [(0, 10), (10, 10), (20, 50), (70, 100)])], [(10, 10), (20, 10), (30, 50), (80, 100)])