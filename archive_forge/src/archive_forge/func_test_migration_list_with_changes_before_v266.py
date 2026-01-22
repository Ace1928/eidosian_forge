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
def test_migration_list_with_changes_before_v266(self):
    self.run_command('migration-list --changes-before 2016-02-29T06:23:22', api_version='2.66')
    self.assert_called('GET', '/os-migrations?changes-before=2016-02-29T06%3A23%3A22')