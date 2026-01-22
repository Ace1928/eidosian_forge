import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_network_trunk_subports_without_optional_keys(self):
    subport = copy.copy(self.new_trunk.sub_ports[0])
    subport.pop('segmentation_type')
    subport.pop('segmentation_id')
    arglist = ['--parent-port', self.new_trunk.port_id, '--subport', 'port=%(port)s' % {'port': subport['port_id']}, self.new_trunk.name]
    verifylist = [('name', self.new_trunk.name), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'port': subport['port_id']}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_trunk.assert_called_once_with(**{'name': self.new_trunk.name, 'admin_state_up': True, 'port_id': self.new_trunk.port_id, 'sub_ports': [subport]})
    self.assertEqual(self.columns, columns)
    data_with_desc = list(self.data)
    data_with_desc[0] = self.new_trunk['description']
    data_with_desc = tuple(data_with_desc)
    self.assertEqual(data_with_desc, data)