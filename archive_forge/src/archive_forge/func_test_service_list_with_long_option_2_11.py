from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_service_list_with_long_option_2_11(self, sm_mock):
    sm_mock.return_value = True
    arglist = ['--host', self.service.host, '--service', self.service.binary, '--long']
    verifylist = [('host', self.service.host), ('service', self.service.binary), ('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.services.assert_called_with(host=self.service.host, binary=self.service.binary)
    columns_long = self.columns_long + ('Forced Down',)
    data_long = [self.data_long[0] + (self.service.is_forced_down,)]
    self.assertEqual(columns_long, columns)
    self.assertEqual(data_long, list(data))