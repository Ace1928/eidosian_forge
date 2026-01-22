from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_with_device_profile(self):
    arglist = ['--network', self._port.network_id, '--device-profile', 'cyborg_device_profile_1', 'test-port']
    verifylist = [('network', self._port.network_id), ('device_profile', self._port.device_profile), ('name', 'test-port')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    create_args = {'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port', 'device_profile': 'cyborg_device_profile_1'}
    self.network_client.create_port.assert_called_once_with(**create_args)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)