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
def test_delete_multi_server_groups(self):
    self.run_command('server-group-delete 12345 56789')
    self.assert_called('DELETE', '/os-server-groups/56789')
    self.assert_called('DELETE', '/os-server-groups/12345', pos=-2)