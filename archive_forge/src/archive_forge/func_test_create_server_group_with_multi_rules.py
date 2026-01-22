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
def test_create_server_group_with_multi_rules(self):
    self.run_command('server-group-create sg1 anti-affinity --rule a=b --rule c=d', api_version='2.64')
    self.assert_called('POST', '/os-server-groups', {'server_group': {'name': 'sg1', 'policy': 'anti-affinity', 'rules': {'a': 'b', 'c': 'd'}}})