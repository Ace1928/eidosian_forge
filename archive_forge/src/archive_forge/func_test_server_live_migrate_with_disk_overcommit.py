import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_server_live_migrate_with_disk_overcommit(self):
    arglist = ['--live-migration', '--disk-overcommit', self.server.id]
    verifylist = [('live_migration', True), ('block_migration', None), ('disk_overcommit', True), ('wait', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_client.api_version = api_versions.APIVersion('2.24')
    result = self.cmd.take_action(parsed_args)
    self.servers_mock.get.assert_called_with(self.server.id)
    self.server.live_migrate.assert_called_with(block_migration=False, disk_over_commit=True, host=None)
    self.assertNotCalled(self.servers_mock.migrate)
    self.assertIsNone(result)