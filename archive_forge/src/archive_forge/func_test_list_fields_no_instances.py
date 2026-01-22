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
@mock.patch('novaclient.tests.unit.v2.fakes.FakeSessionClient.get_servers_detail')
def test_list_fields_no_instances(self, mock_get_servers_detail):
    mock_get_servers_detail.return_value = (200, {}, {'servers': []})
    stdout, _stderr = self.run_command('list --fields metadata,networks')
    defaults = 'ID | Name | Status | Task State | Power State | Networks'
    self.assertIn(defaults, stdout)