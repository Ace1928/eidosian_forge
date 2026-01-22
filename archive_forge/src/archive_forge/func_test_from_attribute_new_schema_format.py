from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
def test_from_attribute_new_schema_format(self):
    s = attributes.Schema('Test description.')
    self.assertIs(s, attributes.Schema.from_attribute(s))
    self.assertEqual('Test description.', attributes.Schema.from_attribute(s).description)
    s = attributes.Schema('Test description.', type=attributes.Schema.MAP)
    self.assertIs(s, attributes.Schema.from_attribute(s))
    self.assertEqual(attributes.Schema.MAP, attributes.Schema.from_attribute(s).type)