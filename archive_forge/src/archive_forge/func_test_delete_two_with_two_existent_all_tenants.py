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
def test_delete_two_with_two_existent_all_tenants(self):
    self.run_command('delete sample-server sample-server2 --all-tenants')
    self.assert_called('GET', '/servers?all_tenants=1&name=sample-server', pos=0)
    self.assert_called('GET', '/servers/1234', pos=1)
    self.assert_called('DELETE', '/servers/1234', pos=2)
    self.assert_called('GET', '/servers?all_tenants=1&name=sample-server2', pos=3)
    self.assert_called('GET', '/servers/5678', pos=4)
    self.assert_called('DELETE', '/servers/5678', pos=5)