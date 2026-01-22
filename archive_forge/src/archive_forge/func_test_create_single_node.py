import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_single_node(self):
    params = {'driver': 'fake'}
    self.client.node.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual(('uuid', None), create_resources.create_single_node(self.client, **params))
    self.client.node.create.assert_called_once_with(driver='fake')