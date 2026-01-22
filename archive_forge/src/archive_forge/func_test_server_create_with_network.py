import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_server_create_with_network(self):
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--network', 'net1', '--nic', 'net-id=net1,v4-fixed-ip=10.0.0.2', '--port', 'port1', '--network', 'net1', '--network', 'auto', '--nic', 'port-id=port2', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', [{'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '10.0.0.2', 'v6-fixed-ip': ''}, {'net-id': '', 'port-id': 'port1', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': 'auto', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': '', 'port-id': 'port2', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}]), ('config_drive', False), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    get_endpoints = mock.Mock()
    get_endpoints.return_value = {'network': []}
    self.app.client_manager.auth_ref = mock.Mock()
    self.app.client_manager.auth_ref.service_catalog = mock.Mock()
    self.app.client_manager.auth_ref.service_catalog.get_endpoints = get_endpoints
    network_resource = mock.Mock(id='net1_uuid')
    port1_resource = mock.Mock(id='port1_uuid')
    port2_resource = mock.Mock(id='port2_uuid')
    self.network_client.find_network.return_value = network_resource
    self.network_client.find_port.side_effect = lambda port_id, ignore_missing: {'port1': port1_resource, 'port2': port2_resource}[port_id]
    _network_1 = mock.Mock(id='net1_uuid')
    _network_auto = mock.Mock(id='auto_uuid')
    _port1 = mock.Mock(id='port1_uuid')
    _port2 = mock.Mock(id='port2_uuid')
    find_network = mock.Mock()
    find_port = mock.Mock()
    find_network.side_effect = lambda net_id, ignore_missing: {'net1': _network_1, 'auto': _network_auto}[net_id]
    find_port.side_effect = lambda port_id, ignore_missing: {'port1': _port1, 'port2': _port2}[port_id]
    self.app.client_manager.network.find_network = find_network
    self.app.client_manager.network.find_port = find_port
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[{'net-id': 'net1_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': 'net1_uuid', 'v4-fixed-ip': '10.0.0.2', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': 'port1_uuid'}, {'net-id': 'net1_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': 'auto_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': 'port2_uuid'}], scheduler_hints={}, config_drive=None)
    self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist(), data)