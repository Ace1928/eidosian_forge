from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_image
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
@mock.patch.object(common_utils, 'wait_for_status', return_value=False)
def test_server_create_image_wait_fail(self, mock_wait_for_status):
    servers = self.setup_servers_mock(count=1)
    images = self.setup_images_mock(count=1, servers=servers)
    arglist = ['--wait', servers[0].id]
    verifylist = [('wait', True), ('server', servers[0].id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.compute_sdk_client.create_server_image.assert_called_with(servers[0].id, servers[0].name, None)
    mock_wait_for_status.assert_called_once_with(self.image_client.get_image, images[0].id, callback=mock.ANY)