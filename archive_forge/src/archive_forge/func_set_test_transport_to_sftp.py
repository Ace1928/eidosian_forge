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
def set_test_transport_to_sftp(testcase):
    """A helper to set transports on test case instances."""
    if getattr(testcase, '_get_remote_is_absolute', None) is None:
        testcase._get_remote_is_absolute = True
    if testcase._get_remote_is_absolute:
        testcase.transport_server = stub_sftp.SFTPAbsoluteServer
    else:
        testcase.transport_server = stub_sftp.SFTPHomeDirServer
    testcase.transport_readonly_server = HttpServer