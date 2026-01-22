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
def test_user_quota_delete(self):
    self.run_command('quota-delete --tenant 97f4c221bff44578b0300df4ef119353 --user u1')
    self.assert_called('DELETE', '/os-quota-sets/97f4c221bff44578b0300df4ef119353?user_id=u1')