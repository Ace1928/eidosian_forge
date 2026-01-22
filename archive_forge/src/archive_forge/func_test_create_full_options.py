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
def test_create_full_options(self):
    self.new_trunk['description'] = 'foo description'
    subport = self.new_trunk.sub_ports[0]
    arglist = ['--disable', '--description', self.new_trunk.description, '--parent-port', self.new_trunk.port_id, '--subport', 'port=%(port)s,segmentation-type=%(seg_type)s,segmentation-id=%(seg_id)s' % {'seg_id': subport['segmentation_id'], 'seg_type': subport['segmentation_type'], 'port': subport['port_id']}, self.new_trunk.name]
    verifylist = [('name', self.new_trunk.name), ('description', self.new_trunk.description), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'port': subport['port_id'], 'segmentation-id': str(subport['segmentation_id']), 'segmentation-type': subport['segmentation_type']}]), ('disable', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_trunk.assert_called_once_with(**{'name': self.new_trunk.name, 'description': self.new_trunk.description, 'admin_state_up': False, 'port_id': self.new_trunk.port_id, 'sub_ports': [subport]})
    self.assertEqual(self.columns, columns)
    data_with_desc = list(self.data)
    data_with_desc[0] = self.new_trunk['description']
    data_with_desc = tuple(data_with_desc)
    self.assertEqual(data_with_desc, data)