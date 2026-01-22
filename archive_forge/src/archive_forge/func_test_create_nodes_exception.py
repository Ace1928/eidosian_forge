import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_portgroups', autospec=True)
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_nodes_exception(self, mock_create_ports, mock_create_portgroups):
    node = {'driver': 'fake', 'ports': ['list of ports'], 'portgroups': ['list of portgroups']}
    self.client.node.create.side_effect = exc.ClientException('bar')
    errs = create_resources.create_nodes(self.client, [node])
    self.assertIsInstance(errs[0], exc.ClientException)
    self.assertEqual(1, len(errs))
    self.client.node.create.assert_called_once_with(driver='fake')
    self.assertFalse(mock_create_ports.called)
    self.assertFalse(mock_create_portgroups.called)