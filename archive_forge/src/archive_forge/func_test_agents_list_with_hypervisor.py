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
def test_agents_list_with_hypervisor(self):
    _, err = self.run_command('agent-list --hypervisor xen')
    self.assert_called('GET', '/os-agents?hypervisor=xen')
    self.assertIn('This command has been deprecated since 23.0.0 Wallaby Release and will be removed in the first major release after the Nova server 24.0.0 X release.', err)