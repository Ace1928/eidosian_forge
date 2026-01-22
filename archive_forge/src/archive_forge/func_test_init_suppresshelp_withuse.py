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
@mock.patch.object(novaclient.shell.argparse.Action, '__init__', return_value=None)
def test_init_suppresshelp_withuse(self, mock_init):
    result = novaclient.shell.DeprecatedAction('option_strings', 'dest', help=argparse.SUPPRESS, use='use this instead', a=1, b=2, c=3)
    self.assertEqual(result.emitted, set())
    self.assertEqual(result.use, 'use this instead')
    self.assertEqual(result.real_action_args, ('option_strings', 'dest', argparse.SUPPRESS, {'a': 1, 'b': 2, 'c': 3}))
    self.assertIsNone(result.real_action)
    mock_init.assert_called_once_with('option_strings', 'dest', help=argparse.SUPPRESS, a=1, b=2, c=3)