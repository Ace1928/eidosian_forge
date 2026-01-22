import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_boot_user_data(self):
    testfile = os.path.join(os.path.dirname(__file__), 'testfile.txt')
    with open(testfile) as testfile_fd:
        data = testfile_fd.read().encode('utf-8')
    expected_file_data = base64.b64encode(data).decode('utf-8')
    self.run_command('boot --flavor 1 --image %s --user-data %s some-server' % (FAKE_UUID_1, testfile))
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1, 'user_data': expected_file_data}})