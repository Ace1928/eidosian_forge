import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import endpoint
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_property_service_url_validate_schema(self):
    schema = endpoint.KeystoneEndpoint.properties_schema[endpoint.KeystoneEndpoint.SERVICE_URL]
    self.assertTrue(schema.update_allowed, 'update_allowed for property %s is modified' % endpoint.KeystoneEndpoint.SERVICE_URL)
    self.assertTrue(schema.required, 'required for property %s is modified' % endpoint.KeystoneEndpoint.SERVICE_URL)
    self.assertEqual(properties.Schema.STRING, schema.type, 'type for property %s is modified' % endpoint.KeystoneEndpoint.SERVICE_URL)
    self.assertEqual('URL of keystone service endpoint.', schema.description, 'description for property %s is modified' % endpoint.KeystoneEndpoint.SERVICE_URL)