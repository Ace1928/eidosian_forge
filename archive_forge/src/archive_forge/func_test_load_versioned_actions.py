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
def test_load_versioned_actions(self):
    parser = novaclient.shell.NovaClientArgumentParser()
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    shell = novaclient.shell.OpenStackComputeShell()
    shell.subcommands = {}
    shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('2.15'), False)
    self.assertIn('fake-action', shell.subcommands.keys())
    self.assertEqual(1, shell.subcommands['fake-action'].get_default('func')())
    parser = novaclient.shell.NovaClientArgumentParser()
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    shell = novaclient.shell.OpenStackComputeShell()
    shell.subcommands = {}
    shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('2.25'), False)
    self.assertIn('fake-action', shell.subcommands.keys())
    self.assertEqual(2, shell.subcommands['fake-action'].get_default('func')())