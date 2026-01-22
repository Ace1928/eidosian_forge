from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_image
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_server_image_create_defaults(self):
    servers = self.setup_servers_mock(count=1)
    images = self.setup_images_mock(count=1, servers=servers)
    arglist = [servers[0].id]
    verifylist = [('server', servers[0].id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_server_image.assert_called_with(servers[0].id, servers[0].name, None)
    self.assertEqual(self.image_columns(images[0]), columns)
    self.assertCountEqual(self.image_data(images[0]), data)