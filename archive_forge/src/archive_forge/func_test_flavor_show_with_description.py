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
def test_flavor_show_with_description(self):
    """Tests that the description is shown in version >= 2.55."""
    out, _ = self.run_command('flavor-show 1', api_version='2.55')
    self.assert_called('GET', '/flavors/1', pos=-2)
    self.assert_called('GET', '/flavors/1/os-extra_specs', pos=-1)
    self.assertIn('description', out)