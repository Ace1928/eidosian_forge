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
def test_flavor_list_with_extra_specs(self):
    self.run_command('flavor-list --extra-specs')
    self.assert_called('GET', '/flavors/aa1/os-extra_specs')
    self.assert_called_anytime('GET', '/flavors/detail')