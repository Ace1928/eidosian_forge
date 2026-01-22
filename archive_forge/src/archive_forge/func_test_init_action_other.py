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
@mock.patch.object(argparse.Action, '__init__', return_value=None)
def test_init_action_other(self, mock_init):
    action = mock.Mock()
    result = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action=action, a=1, b=2, c=3)
    self.assertEqual(result.emitted, set())
    self.assertIsNone(result.use)
    self.assertIs(result.real_action_args, False)
    self.assertEqual(result.real_action, action.return_value)
    mock_init.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)
    action.assert_called_once_with('option_strings', 'dest', help='Deprecated', a=1, b=2, c=3)