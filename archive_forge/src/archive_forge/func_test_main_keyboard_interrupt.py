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
@mock.patch.object(novaclient.shell.OpenStackComputeShell, 'main')
def test_main_keyboard_interrupt(self, mock_compute_shell):
    mock_compute_shell.side_effect = KeyboardInterrupt()
    try:
        novaclient.shell.main([])
    except SystemExit as ex:
        self.assertEqual(ex.code, 130)