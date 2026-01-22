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
def test_instance_action_list_with_changes_since_invalid_value_v258(self):
    ex = self.assertRaises(exceptions.CommandError, self.run_command, 'instance-action-list sample-server --changes-since 0123456789', api_version='2.58')
    self.assertIn('Invalid changes-since value', str(ex))