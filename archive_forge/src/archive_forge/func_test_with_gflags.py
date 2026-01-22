import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def test_with_gflags(self):
    HOST = 'myhostname'
    PORT = '144169'

    class MockFlags(object):
        auth_host_name = HOST
        auth_host_port = PORT
        auth_local_webserver = False
    credentials_lib.FLAGS = MockFlags
    flags = credentials_lib._GetRunFlowFlags(['--auth_host_name=%s' % HOST, '--auth_host_port=%s' % PORT, '--noauth_local_webserver'])
    self.assertEqual(flags.auth_host_name, HOST)
    self.assertEqual(flags.auth_host_port, PORT)
    self.assertEqual(flags.logging_level, 'ERROR')
    self.assertEqual(flags.noauth_local_webserver, True)