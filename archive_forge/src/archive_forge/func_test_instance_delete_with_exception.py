from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch('troveclient.utils.get_resource_id_by_name')
def test_instance_delete_with_exception(self, mock_getid):
    mock_getid.side_effect = exceptions.CommandError
    args = ['fakeinstance']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)