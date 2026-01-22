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
def test_flavor_list_with_extra_specs_2_61_or_later(self):
    """Tests that the 'os-extra_specs' API is not called
        when the '--extra-specs' option is specified since microversion 2.61.
        """
    out, _ = self.run_command('flavor-list --extra-specs', api_version='2.61')
    self.assert_not_called('GET', '/flavors/aa1/os-extra_specs')
    self.assert_called_anytime('GET', '/flavors/detail')
    self.assertIn('extra_specs', out)