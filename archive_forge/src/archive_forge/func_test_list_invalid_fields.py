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
def test_list_invalid_fields(self):
    self.assertRaises(exceptions.CommandError, self.run_command, 'list --fields host,security_groups,OS-EXT-MOD:some_thing,invalid')
    self.assertRaises(exceptions.CommandError, self.run_command, 'list --fields __dict__')
    self.assertRaises(exceptions.CommandError, self.run_command, 'list --fields update')
    self.assertRaises(exceptions.CommandError, self.run_command, 'list --fields __init__')
    self.assertRaises(exceptions.CommandError, self.run_command, 'list --fields __module__,updated')