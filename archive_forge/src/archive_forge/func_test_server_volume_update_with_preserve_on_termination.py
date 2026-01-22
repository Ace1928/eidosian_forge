from unittest import mock
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_volume
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_server_volume_update_with_preserve_on_termination(self, sm_mock):
    sm_mock.return_value = True
    arglist = [self.server.id, self.volume.id, '--preserve-on-termination']
    verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.update_volume_attachment.assert_called_once_with(self.server, self.volume, delete_on_termination=False)
    self.assertIsNone(result)