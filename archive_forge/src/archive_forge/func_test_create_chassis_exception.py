import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_nodes', autospec=True)
def test_create_chassis_exception(self, mock_create_nodes):
    chassis = {'description': 'fake', 'nodes': ['list of nodes']}
    self.client.chassis.create.side_effect = exc.ClientException('bar')
    errs = create_resources.create_chassis(self.client, [chassis])
    self.client.chassis.create.assert_called_once_with(description='fake')
    self.assertFalse(mock_create_nodes.called)
    self.assertEqual(1, len(errs))
    self.assertIsInstance(errs[0], exc.ClientException)