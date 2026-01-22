from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch.object(utils, 'find_resource')
def test_instance_resize_flavor(self, mock_find):
    args = ['instance1', 'flavor_id']
    mock_find.side_effect = [mock.MagicMock(id='fake_instance_id'), mock.MagicMock(id='fake_flavor_id')]
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.instance_client.resize_instance.assert_called_with('fake_instance_id', 'fake_flavor_id')