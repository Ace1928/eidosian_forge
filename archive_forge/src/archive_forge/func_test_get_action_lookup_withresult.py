import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
@mock.patch.object(sys, 'stderr', io.StringIO())
def test_get_action_lookup_withresult(self):
    action_class = mock.Mock()
    parser = mock.Mock(**{'_registry_get.return_value': action_class})
    obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='store', const=1)
    result = obj._get_action(parser)
    self.assertEqual(result, action_class.return_value)
    self.assertEqual(obj.real_action, action_class.return_value)
    parser._registry_get.assert_called_once_with('action', 'store')
    action_class.assert_called_once_with('option_strings', 'dest', help='Deprecated', const=1)
    self.assertEqual(sys.stderr.getvalue(), '')