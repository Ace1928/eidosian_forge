from unittest import mock
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_volume
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_server_volume_list_with_tags(self, sm_mock):
    sm_mock.side_effect = [False, True, False, False]
    arglist = [self.server.id]
    verifylist = [('server', self.server.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(('ID', 'Device', 'Server ID', 'Volume ID', 'Tag'), columns)
    self.assertEqual(((self.volume_attachments[0].id, self.volume_attachments[0].device, self.volume_attachments[0].server_id, self.volume_attachments[0].volume_id, self.volume_attachments[0].tag), (self.volume_attachments[1].id, self.volume_attachments[1].device, self.volume_attachments[1].server_id, self.volume_attachments[1].volume_id, self.volume_attachments[1].tag)), tuple(data))
    self.compute_sdk_client.volume_attachments.assert_called_once_with(self.server)