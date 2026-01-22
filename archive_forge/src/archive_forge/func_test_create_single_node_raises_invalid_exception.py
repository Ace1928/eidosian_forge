import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_single_node_raises_invalid_exception(self):
    params = {'driver': 'fake'}
    e = exc.InvalidAttribute('foo')
    self.client.node.create.side_effect = e
    res, err = create_resources.create_single_node(self.client, **params)
    self.assertIsNone(res)
    self.assertIsInstance(err, exc.InvalidAttribute)
    self.assertIn('Cannot create the node with attributes', str(err))
    self.client.node.create.assert_called_once_with(driver='fake')