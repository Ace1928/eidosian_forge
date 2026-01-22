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
def test_rebuild_invalid_user_data(self):
    invalid_file = os.path.join(os.path.dirname(__file__), 'no_such_file')
    cmd = 'rebuild sample-server %s --user-data %s' % (FAKE_UUID_1, invalid_file)
    ex = self.assertRaises(exceptions.CommandError, self.run_command, cmd, api_version='2.57')
    self.assertIn("Can't open '%(user_data)s': [Errno 2] No such file or directory: '%(user_data)s'" % {'user_data': invalid_file}, str(ex))