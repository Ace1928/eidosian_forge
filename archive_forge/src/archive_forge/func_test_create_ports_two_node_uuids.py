import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_ports_two_node_uuids(self):
    port = {'address': 'fake-address', 'node_uuid': 'node-uuid-1'}
    errs = create_resources.create_ports(self.client, [port], 'node-uuid-2')
    self.assertIsInstance(errs[0], exc.ClientException)
    self.assertEqual(1, len(errs))
    self.assertFalse(self.client.port.create.called)