import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_single_portgroup_with_ports(self):
    params = {'ports': ['some ports'], 'node_uuid': 'fake-node-uuid'}
    self.client.portgroup.create.return_value = mock.Mock(uuid='fake-portgroup-uuid')
    self.assertEqual(('fake-portgroup-uuid', None), create_resources.create_single_portgroup(self.client, **params))
    self.client.portgroup.create.assert_called_once_with(node_uuid='fake-node-uuid')