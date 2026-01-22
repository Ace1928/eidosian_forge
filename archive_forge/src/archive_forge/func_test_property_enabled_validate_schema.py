from unittest import mock
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.keystone import project
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_property_enabled_validate_schema(self):
    schema = project.KeystoneProject.properties_schema[project.KeystoneProject.ENABLED]
    self.assertEqual(True, schema.update_allowed, 'update_allowed for property %s is modified' % project.KeystoneProject.DOMAIN)
    self.assertEqual(properties.Schema.BOOLEAN, schema.type, 'type for property %s is modified' % project.KeystoneProject.ENABLED)
    self.assertEqual('This project is enabled or disabled.', schema.description, 'description for property %s is modified' % project.KeystoneProject.ENABLED)
    self.assertEqual(True, schema.default, 'default for property %s is modified' % project.KeystoneProject.ENABLED)