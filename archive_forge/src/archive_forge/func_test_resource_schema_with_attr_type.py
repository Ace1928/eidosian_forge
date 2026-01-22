from unittest import mock
from heat.common import exception
from heat.engine import environment
from heat.engine import resource as res
from heat.engine import service
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_resource_schema_with_attr_type(self):
    type_name = 'ResourceWithAttributeType'
    expected = {'resource_type': type_name, 'properties': {}, 'attributes': {'attr1': {'description': 'A generic attribute', 'type': 'string'}, 'attr2': {'description': 'Another generic attribute', 'type': 'map'}, 'show': {'description': 'Detailed information about resource.', 'type': 'map'}}, 'support_status': {'status': 'SUPPORTED', 'version': None, 'message': None, 'previous_status': None}}
    schema = self.eng.resource_schema(self.ctx, type_name=type_name)
    self.assertEqual(expected, schema)