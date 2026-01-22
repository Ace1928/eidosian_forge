import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_bad_connection_ssh(self):
    """None => auto-detect vendor"""
    f = open(os.devnull, 'wb')
    self.addCleanup(f.close)
    self.set_vendor(None, f)
    t = _mod_transport.get_transport_from_url(self.bogus_url)
    try:
        self.assertRaises(errors.ConnectionError, t.get, 'foobar')
    except NameError as e:
        if "global name 'SSHException'" in str(e):
            self.knownFailure('Known NameError bug in paramiko 1.6.1')
        raise