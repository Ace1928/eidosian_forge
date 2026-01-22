import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import service
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_property_type_validate_schema(self):
    schema = service.KeystoneService.properties_schema[service.KeystoneService.TYPE]
    self.assertTrue(schema.update_allowed, 'update_allowed for property %s is modified' % service.KeystoneService.TYPE)
    self.assertTrue(schema.required, 'required for property %s is modified' % service.KeystoneService.TYPE)
    self.assertEqual(properties.Schema.STRING, schema.type, 'type for property %s is modified' % service.KeystoneService.TYPE)
    self.assertEqual('Type of keystone Service.', schema.description, 'description for property %s is modified' % service.KeystoneService.TYPE)